import torch
import torch.nn as nn
from neuronav.agents.base_agent import BaseAgent
from neuronav.utils import discount_features


class PPOAgent(BaseAgent):
    def __init__(
        self,
        obs_size,
        act_size,
        lr=3e-4,
        gamma=0.99,
        poltype="softmax",
        beta=1.0,
        epsilon=1e-10,
        max_buffer_size=100,
        entropy_bonus=1e-3,
        grad_steps=3,
    ):
        super().__init__(obs_size, act_size, lr, gamma, poltype, beta, epsilon)
        self.policy_network = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.GELU(),
            nn.Linear(64, act_size),
            nn.Softmax(dim=-1),
        )
        self.value_network = nn.Sequential(
            nn.Linear(obs_size, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.max_buffer_size = max_buffer_size
        self.entropy_bonus = entropy_bonus
        self.grad_steps = grad_steps
        self.buffer = ExperienceBuffer()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), self.lr)
        self.reset()

    def reset(self):
        self.buffer.clear()

    def sample_action(self, state):
        state = self.linear_prepare_state(state)
        state = torch.tensor(state).unsqueeze(0).type(torch.float32)
        logits = self.policy_network(state)[0]
        value_est = self.value_network(state)[0]
        self.buffer.p.append(logits)
        self.buffer.v.append(value_est)
        return self.base_sample_action(torch.log(logits + 1e-10).detach().numpy())

    def _update(self, current_exp):
        s = self.linear_prepare_state(current_exp[0])
        s_a = current_exp[1]
        s_1 = self.linear_prepare_state(current_exp[2])
        r = current_exp[3]
        d = current_exp[4]
        self.buffer.append(s, s_a, r, s_1, d)
        if d:
            for _ in range(self.grad_steps):
                self.update_model()
            self.reset()

    def update_model(self):
        self.optimizer.zero_grad()
        x, a, p, r, x1, d, v = self.buffer.sample()
        x = x.type(torch.float32)
        p_new = self.policy_network(x)
        v_new = self.value_network(x)
        v = v.squeeze(-1)
        if d[-1]:
            returns = discount_features(r, self.gamma)
        else:
            v_next = (
                self.value_network(x1[-1:].type(torch.float32)).squeeze(-1).detach()
            )
            returns = discount_features(r, self.gamma, v_next)
        advantages = returns - v.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()
        entropy = -(torch.log(p_new + 1e-10) * p_new).mean()
        p_old = p[:, a].diag()
        p_new = p_new[:, a].diag()
        ratio = p_new / p_old.detach()
        clip_ratio = torch.clip(ratio, 1 - 0.2, 1 + 0.2)
        policy_loss = -torch.minimum(ratio * advantages, clip_ratio * advantages).mean()
        value_loss = (returns - v_new).pow(2).mean()
        loss = policy_loss + value_loss - self.entropy_bonus * entropy
        loss.backward()
        self.optimizer.step()


class ExperienceBuffer:
    def __init__(self):
        self.clear()

    def append(self, x, a, r, x1, d):
        self.x.append(torch.tensor(x))
        self.a.append(torch.tensor(a))
        self.r.append(torch.tensor(r))
        self.x1.append(torch.tensor(x1))
        self.d.append(torch.tensor(d))

    @property
    def len(self):
        return len(self.x)

    def clear(self):
        self.x = []
        self.a = []
        self.p = []
        self.r = []
        self.x1 = []
        self.d = []
        self.v = []

    def sample(self):
        x = torch.stack(self.x)
        a = torch.stack(self.a)
        p = torch.stack(self.p)
        r = torch.stack(self.r)
        x1 = torch.stack(self.x1)
        d = torch.stack(self.d)
        v = torch.stack(self.v)
        return x, a, p, r, x1, d, v
