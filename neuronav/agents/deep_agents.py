import torch
import torch.nn as nn
from neuronav.agents.base_agent import BaseAgent


class PGAgent(BaseAgent):
    def __init__(
        self,
        obs_size,
        act_size,
        lr=3e-4,
        gamma=0.99,
        poltype="softmax",
        beta=1.0,
        epsilon=1e-10,
    ):
        super().__init__(obs_size, act_size, lr, gamma, poltype, beta, epsilon)
        self.network = PGNetwork(obs_size, act_size)
        self.buffer = ExperienceBuffer()

    def sample_action(self, state):
        state = self.linear_prepare_state(state)
        state = torch.tensor(state).unsqueeze(0).type(torch.float32)
        logits = self.network(state)
        return self.base_sample_action(logits.detach().numpy())

    def _update(self, current_exp):
        s = self.linear_prepare_state(current_exp[0])
        s_a = current_exp[1]
        s_1 = self.linear_prepare_state(current_exp[2])
        r = current_exp[3]
        d = current_exp[4]
        self.buffer.append(s, s_a, r, s_1, d)
        if d:
            self.buffer.clear()


class PGNetwork(nn.Module):
    def __init__(self, obs_size, act_size, h_size=64):
        super(PGNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_size, h_size), nn.GELU(), nn.Linear(h_size, act_size)
        )

    def forward(self, obs):
        return self.layers(obs)


class ExperienceBuffer:
    def __init__(self):
        self.clear()

    def append(self, x, a, r, x1, d):
        self.x.append(x)
        self.a.append(a)
        self.r.append(r)
        self.x1.append(x1)
        self.d.append(d)

    def clear(self):
        self.x = []
        self.a = []
        self.r = []
        self.x1 = []
        self.d = []

    def sample_buffer(self):
        x = torch.stack(self.x)
        a = torch.stack(self.a)
        r = torch.stack(self.r)
        x1 = torch.stack(self.x1)
        d = torch.stack(self.d)
        return x, a, r, x1, d
