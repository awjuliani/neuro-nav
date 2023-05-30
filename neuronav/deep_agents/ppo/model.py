import torch
import torch.nn as nn
import torch.optim as optim
from neuronav.deep_agents.modules import gen_encoder


class PPOModel(torch.nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.enc_type = model_params["enc_type"]
        self.obs_size = model_params["obs_size"]
        self.act_size = model_params["act_size"]
        self.h_size = model_params["h_size"]
        self.depth = model_params["depth"]
        self.encoder = gen_encoder(
            self.obs_size, self.h_size, self.depth, self.enc_type
        )
        self.policy = nn.Linear(self.h_size, self.act_size)
        self.value = nn.Linear(self.h_size, 1)
        self.optimizer = optim.AdamW(self.parameters(), lr=model_params["lr"])

    def forward(self, x):
        x = x.view(-1, self.obs_size)
        h = self.encoder(x)
        logits = self.policy(h)
        value = self.value(h)
        return logits, value.view(-1)

    def sample_action(self, obs):
        logits, value = self.forward(obs)
        action = torch.distributions.Categorical(logits=logits).sample()
        return action, logits, value.view(-1)
