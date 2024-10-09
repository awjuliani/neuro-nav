import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from neuronav.deep_agents.agent import BaseAgent
from neuronav.deep_agents.ppo.model import PPOModel

EPSILON = 1e-8


class PPOAgent(BaseAgent):
    def __init__(
        self,
        model_params,
        agent_params,
    ):
        model = PPOModel(model_params)
        super().__init__(model, agent_params)
        self.gamma = agent_params["gamma"]
        self.lamda = agent_params["lambda"]
        self.buffer_size = agent_params["buffer_size"]
        self.ent_coef = agent_params["ent_coef"]
        self.num_passes = agent_params["num_passes"]
        self.clip_param = agent_params["clip_param"]
        self.grad_clip = agent_params["grad_clip"]
        self.reset_buffer()

    def sample_action(self, obs):
        action, _, _ = self.model.sample_action(obs)
        return action

    def sample_value(self, obs):
        _, value = self.model(obs)
        return value

    def sample_policy(self, obs):
        logits, _ = self.model(obs)
        return F.softmax(logits, dim=-1)

    def sample_hidden(self, obs):
        return self.model.encode(obs)

    def reset(self):
        if len(self.ep_obs) > 0:
            self.append_buffer()

    def append_buffer(self):
        self.buffer_logits.extend(self.ep_logits)
        self.buffer_values.extend(self.ep_values)
        ep_values = list(torch.stack(self.ep_values).detach())
        ep_rewards = self.gae(self.ep_rewards, ep_values, 0.0, self.ep_dones)
        self.buffer_obs.extend(self.ep_obs)
        self.buffer_actions.extend(self.ep_acts)
        self.buffer_advantages.extend(ep_rewards)
        self.reset_ep()

    def reset_buffer(self):
        self.buffer_obs = []
        self.buffer_actions = []
        self.buffer_logits = []
        self.buffer_values = []
        self.buffer_advantages = []
        self.epoch_train_returns = []
        self.buffer_lengths = []
        self.reset_ep()

    def reset_ep(self):
        self.ep_obs = []
        self.ep_acts = []
        self.ep_rewards = []
        self.ep_dones = []
        self.ep_logits = []
        self.ep_values = []

    def update(self, current_exp):
        obs, action, obs_next, reward, done = current_exp
        reward = torch.tensor(reward, dtype=torch.float32)
        self.ep_obs.append(obs)
        self.ep_acts.append(action)
        self.ep_rewards.append(reward)
        self.ep_dones.append(done)
        logits, value = self.model(obs)
        self.ep_logits.append(logits)
        self.ep_values.append(value)
        if done:
            self.append_buffer()
        if len(self.buffer_obs) > self.buffer_size:
            buffer_obs = torch.stack(self.buffer_obs)
            buffer_actions = torch.tensor(self.buffer_actions)
            buffer_advantages = torch.stack(self.buffer_advantages).squeeze(1)
            buffer_logits = torch.stack(self.buffer_logits).squeeze(1)
            buffer_values = torch.stack(self.buffer_values).squeeze(1)
            self.update_model(
                buffer_obs,
                buffer_actions,
                buffer_advantages,
                buffer_logits,
                buffer_values,
            )
            self.reset_buffer()

    def update_model(
        self,
        buffer_obs,
        buffer_actions,
        buffer_advantages,
        buffer_logits,
        buffer_values,
    ):
        buffer_advantages = buffer_advantages.to(self.device).detach()
        buffer_actions = buffer_actions.to(self.device).detach()
        value_targets = buffer_advantages + buffer_values.detach()
        old_log_probs = F.log_softmax(buffer_logits, dim=-1).detach()
        old_log_probs = old_log_probs.gather(1, buffer_actions.unsqueeze(1)).squeeze(1)

        total_pg_loss = []
        total_v_loss = []
        total_v_error = []
        total_e_loss = []
        total_grad_norm = []

        # Normalize advantages once outside the loop
        buffer_advantages = (buffer_advantages - buffer_advantages.mean()) / (
            buffer_advantages.std() + EPSILON
        )

        for _ in range(self.num_passes):
            # Shuffle the data and iterate over minibatches
            minibatch = torch.randperm(len(buffer_obs))
            for i in range(0, len(buffer_obs), self.batch_size):
                batch = minibatch[i : i + self.batch_size]
                if len(batch) < self.batch_size:
                    continue
                batch_obs = buffer_obs[batch]
                batch_actions = buffer_actions[batch]
                batch_value_target = value_targets[batch]
                batch_adv = buffer_advantages[batch]
                batch_old_log_probs = old_log_probs[batch]
                batch_logits, batch_values = self.model(batch_obs.to(self.device))
                batch_new_log_probs = F.log_softmax(batch_logits, dim=-1)
                batch_new_log_probs = batch_new_log_probs.gather(
                    1, batch_actions.unsqueeze(1)
                ).squeeze(1)
                batch_entropy = dist.Categorical(logits=batch_logits).entropy().mean()
                # compute the clipped policy loss
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)
                clip_ratio = torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surr1 = ratio * batch_adv
                surr2 = clip_ratio * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                # compute the value loss
                value_loss = F.mse_loss(batch_values, batch_value_target)
                with torch.no_grad():
                    value_error = (batch_value_target - batch_values).mean()

                loss = (
                    policy_loss + 0.5 * value_loss - self.ent_coef * batch_entropy
                )  # Simplified entropy term
                self.model.optimizer.zero_grad()
                loss.backward()
                # clip the gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )  # Use grad_clip
                self.model.optimizer.step()
                total_pg_loss.append(policy_loss.item())
                total_v_loss.append(value_loss.item())
                total_e_loss.append(batch_entropy.item())
                total_grad_norm.append(grad_norm.item())
                total_v_error.append(value_error.item())
        pg_loss = np.mean(total_pg_loss)
        v_loss = np.mean(total_v_loss)
        e_loss = np.mean(total_e_loss)
        grad_norm = np.mean(total_grad_norm)
        v_error = np.mean(total_v_error)
        return pg_loss, v_loss, e_loss, grad_norm, v_error

    def gae(self, rewards, values, next_value, dones):
        # generalized advantage estimation
        values = values + [next_value]
        gae = 0
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step]
                + self.gamma * values[step + 1] * (1 - dones[step])
                - values[step]
            )
            gae = delta + self.gamma * self.lamda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        return advantages
