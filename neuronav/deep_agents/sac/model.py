import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from neuronav.deep_agents.modules import gen_encoder


class SACModel(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.policy = SACPolicy(model_params)
        self.critic = SACCritic(model_params)

    def forward(self, x):
        return self.policy(x), self.critic(x)


class SACPolicy(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.enc_type = model_params["enc_type"]
        self.obs_size = model_params["obs_size"]
        self.act_size = model_params["act_size"]
        self.h_size = model_params["h_size"]
        self.depth = model_params["depth"]
        self.lr = model_params["lr"]
        self.encoder = gen_encoder(
            self.obs_size, self.h_size, self.depth, self.enc_type
        )
        self.policy = nn.Linear(self.h_size, self.act_size)
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.view(-1, self.obs_size)
        pol_h = self.encoder(x)
        logits = self.policy(pol_h)
        return logits

    def sample_action(self, obs):
        logits = self.forward(obs)
        action = torch.distributions.Categorical(logits=logits).sample()
        return action, logits


class SACCritic(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.enc_type = model_params["enc_type"]
        self.obs_size = model_params["obs_size"]
        self.act_size = model_params["act_size"]
        self.h_size = model_params["h_size"]
        self.depth = model_params["depth"]
        self.lr = model_params["lr"]
        self.critic1_encoder = gen_encoder(
            self.obs_size, self.h_size, self.depth, self.enc_type
        )
        self.critic1 = nn.Linear(self.h_size, self.act_size)
        self.critic2_encoder = gen_encoder(
            self.obs_size, self.h_size, self.depth, self.enc_type
        )
        self.critic2 = nn.Linear(self.h_size, self.act_size)
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

    def forward(self, obs):
        x = obs.view(-1, self.obs_size)
        c1_h = self.critic1_encoder(x)
        value1 = self.critic1(c1_h)
        c2_h = self.critic2_encoder(x)
        value2 = self.critic2(c2_h)
        return value1, value2

    def copy(self):
        return copy.deepcopy(self)
