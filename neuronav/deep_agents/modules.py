import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def gen_encoder(obs_size, h_size, depth, enc_type):
    if enc_type == "conv32":
        return ConvEncoder(h_size, depth, 32)
    elif enc_type == "conv64":
        return ConvEncoder(h_size, depth, 64)
    elif enc_type == "linear":
        return LinearEncoder(obs_size, h_size)
    else:
        raise NotImplementedError


def discount_rewards(rewards, dones, gamma=0.99, v_next=0.0):
    # discount future rewards back to the present using gamma and v_next
    discounted_rewards = torch.zeros_like(rewards)
    R = v_next
    for t in reversed(range(rewards.shape[1])):
        R = rewards[:, t] + gamma * R * (1 - dones[:, t])
        discounted_rewards[:, t] = R
    return discounted_rewards


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LnGelu(nn.Module):
    # Simple layernorm followed by GELU
    def __init__(self, h_size):
        super().__init__()
        self.norm = nn.LayerNorm(h_size)

    def forward(self, x):
        return F.gelu(self.norm(x))


class GnGeLU(nn.Module):
    # Simple groupnorm followed by GELU
    def __init__(self, h_size, groups=1):
        super().__init__()
        self.norm = nn.GroupNorm(groups, h_size)

    def forward(self, x):
        return F.gelu(self.norm(x))


class ConvEncoder(nn.Module):
    def __init__(self, h_size, depth, conv_size):
        super().__init__()
        self.depth = depth
        self.h_size = h_size
        self.encoder = self.conv32() if conv_size == 32 else self.conv64()

    def conv32(self):
        return nn.Sequential(
            nn.Unflatten(1, (self.depth, 32, 32)),
            nn.Conv2d(self.depth, 16, 4, 2, 1),
            GnGeLU(16),
            nn.Conv2d(16, 32, 4, 2, 1),
            GnGeLU(32),
            nn.Conv2d(32, 64, 4, 2, 1),
            GnGeLU(64),
            nn.Flatten(1, -1),
            nn.Linear(1024, self.h_size),
            LnGelu(self.h_size),
        )

    def conv64(self):
        return nn.Sequential(
            nn.Unflatten(1, (self.depth, 64, 64)),
            nn.Conv2d(self.depth, 32, 4, 2, 1),
            GnGeLU(32),
            nn.Conv2d(32, 64, 4, 2, 1),
            GnGeLU(64),
            nn.Conv2d(64, 128, 4, 2, 1),
            GnGeLU(128),
            nn.Conv2d(128, 128, 4, 2, 1),
            GnGeLU(128),
            nn.Flatten(1, -1),
            nn.Linear(2048, self.h_size),
            LnGelu(self.h_size),
        )

    def forward(self, x):
        return self.encoder(x)


class LinearEncoder(nn.Module):
    def __init__(self, obs_size, h_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, h_size),
            LnGelu(h_size),
            nn.Linear(h_size, h_size),
            LnGelu(h_size),
        )

    def forward(self, x):
        return self.encoder(x)


class ReplayBuffer(object):
    # Replay buffer for SAC
    def __init__(self, max_size=1e6):
        self.max_size = max_size
        self.reset()

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        # x, y, a, r, d represent state, next_state, action, reward, done
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, a, r, d = [], [], [], [], []

        for i in ind:
            X, Y, A, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            a.append(np.array(A, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return (
            torch.FloatTensor(np.array(x)),
            torch.FloatTensor(np.array(y)),
            torch.FloatTensor(np.array(a)),
            torch.FloatTensor(np.array(r)).unsqueeze(1),
            torch.FloatTensor(np.array(d)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.storage)

    def reset(self):
        self.storage = []
        self.ptr = 0
