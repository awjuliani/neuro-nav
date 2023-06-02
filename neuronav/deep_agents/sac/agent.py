import torch
import torch.nn.functional as F
import numpy as np
from neuronav.deep_agents.agent import BaseAgent
from neuronav.deep_agents.modules import ReplayBuffer
from neuronav.deep_agents.sac.model import SACModel

EPSILON = 1e-8


class SACAgent(BaseAgent):
    def __init__(self, model_params, agent_params):
        model = SACModel(model_params)
        super(SACAgent, self).__init__(model, agent_params)
        self.gamma = agent_params["gamma"]
        self.tau = agent_params["tau"]
        self.target_update_interval = agent_params["target_update_interval"]
        self.replay_buffer_size = agent_params["replay_buffer_size"]
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.update_interval = agent_params["update_interval"]
        self.warmup_steps = agent_params["warmup_steps"]
        self.critic_target = self.model.critic.copy()
        self.critic_target.to(self.device)
        self.alpha = agent_params["alpha"]
        self.total_steps = 0
        self.total_updates = 0

    def update_parameters(self):
        # Sample a batch from replay buffer
        x, y, a, r, d = self.replay_buffer.sample(self.batch_size)

        x = x.to(self.device)
        y = y.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        d = d.to(self.device)

        # Compute the target Q value
        with torch.no_grad():
            _, log_pi = self.model.policy.sample_action(y)
            target_Q1, target_Q2 = self.critic_target(y)
            q = torch.mean(torch.stack([target_Q1, target_Q2]), dim=0)
            probs = F.softmax(log_pi, dim=-1)
            lp = torch.log(probs + EPSILON)
            target_V = (probs * (q - self.alpha * lp)).sum(dim=-1).unsqueeze(-1)
            target_Q = r + (1 - d) * self.gamma * target_V
            target_Q = target_Q.detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.model.critic(x)
        a_batch = a.unsqueeze(-1).long()
        current_Q1 = current_Q1.gather(1, a_batch)
        current_Q2 = current_Q2.gather(1, a_batch)

        # Compute critic loss
        critic_loss = 0.5 * F.mse_loss(current_Q1, target_Q) + 0.5 * F.mse_loss(
            current_Q2, target_Q
        )

        with torch.no_grad():
            v_error = (
                0.5 * (target_Q - current_Q1).mean()
                + 0.5 * (target_Q - current_Q2).mean()
            )

        # Optimize the critic
        self.model.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.model.critic.optimizer.step()

        # Compute actor loss
        _, log_pi = self.model.policy.sample_action(x)
        q1, q2 = self.model.critic(x)
        q = torch.mean(torch.stack([q1, q2]), dim=0).detach()
        probs = F.softmax(log_pi, dim=-1)
        lp = torch.log(probs + EPSILON)
        entropy = -(probs * lp).sum(-1)
        exp_q = (probs * q).sum(-1)
        actor_loss = (-exp_q - self.alpha * entropy).mean()

        # Optimize the actor
        self.model.policy.optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), 1.0)
        self.model.policy.optimizer.step()

        # Update the frozen target models
        if self.total_updates % self.target_update_interval == 0:
            self.update_target()
        self.total_updates += 1
        return critic_loss.item(), actor_loss.item(), grad_norm.item(), v_error.item()

    def update_target(self):
        for param, target_param in zip(
            self.model.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def reset(self):
        pass

    def update(self, current_exp):
        state, action, next_state, reward, done = current_exp
        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)
        self.total_steps += 1
        self.replay_buffer.add((state, next_state, action, reward, done))
        if (
            self.total_steps > self.warmup_steps
            and self.total_steps % self.update_interval == 0
        ):
            self.update_parameters()
            self.total_updates += 1

    def sample_action(self, obs):
        obs = torch.Tensor(obs)
        action, log_pi = self.model.policy.sample_action(
            obs.unsqueeze(0).to(self.device)
        )
        action = action.item()
        return action

    def sample_value(self, obs):
        obs = torch.Tensor(obs)
        value = self.model.critic(obs.unsqueeze(0).to(self.device))
        value = value.item()
        return value

    def sample_policy(self, obs):
        obs = torch.Tensor(obs)
        action, log_pi = self.model.policy.sample_action(
            obs.unsqueeze(0).to(self.device)
        )
        probs = F.softmax(log_pi, dim=-1)
        probs = probs.squeeze().detach().cpu().numpy()
        return probs

    def sample_hidden(self, obs):
        obs = torch.Tensor(obs)
        hidden_a, _ = self.model.critic.encode(obs.unsqueeze(0).to(self.device))
        hidden_a = hidden_a.squeeze().detach().cpu().numpy()
        return hidden_a
