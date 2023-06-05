# Deep Reinforcement Learning Algorithms

A set of deep reinforcement learning algorithms implemented in PyTorch.

## Included Algorithms

| Algorithm | Function(s) | Reference | Description | Code Link |
| --- | --- | --- | --- | --- |
| PPO | V(s), π(a \| s) | [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) | A proximal policy gradient algorithm (on-policy) | [Code](./ppo/) |
| SAC | Q(s, a), π(a \| s) | [Haarnoja et al., 2018](https://arxiv.org/abs/1812.05905) | A maximum entropy soft actor-critic algorithm (off-policy) | [Code](./sac/) |

## Hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| `batch_size` | 32 | Number of samples per training batch |
| `learning_rate` | 3e-4 | Learning rate for the optimizer |
| `h_size` | 128 | Size of the hidden layers used in the neural network |

### PPO Hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| `gamma` | 0.99 | Discount factor of the return. Higher values consider longer-term rewards. |
| `lamda` | 0.95 | GAE mixing parameter. Determines how much to rely on value estimates or rollouts when updating policy. |
| `clip_param` | 0.2 | PPO clipping parameter. Determines how conservative updates should be |
| `buffer_size` | 256 | Number of samples to collect before updating the policy |
| `num_passes` | 2 | Number of update passes to make over the collected samples |
| `ent_coef` | 0.02 | Determines how much to bias policy updates toward maximum entropy |

### SAC Hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| `gamma` | 0.99 | Discount factor of the return. Higher values consider longer-term rewards. |
| `tau` | 0.005 | Mixing parameter to use when updating the target network |
| `alpha` | 0.2 | Determines how much to bias policy updates toward maximum entropy |
| `target_update_interval` | 2 | Number of environment steps between target network updates |
| `replay_buffer_size` | 1000000 | Maximum size of the replay buffer |
| `warmup_steps` | 1000 | Number of experience steps to collect before training |
| `update_interval` | 4 | Number of environment steps between training updates |
