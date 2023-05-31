# Deep Reinforcement Learning Algorithms

A set of deep reinforcement learning algorithms implemented in PyTorch.

## Included Algorithms

| Algorithm | Reference | Description | Code Link |
| --- | --- | --- | --- |
| PPO | [Schulman et al. 2017](https://arxiv.org/abs/1707.06347) | A proximal policy gradient algorithm | [Code](./ppo/) |
| SAC | [Haarnoja et al., 2018](https://arxiv.org/abs/1812.05905) | A maximum entropy soft actor-critic algorithm | [Code](./sac/) |

## Hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| `batch_size` | 32 | Number of samples per training batch |
| `learning_rate` | 3e-4 | Learning rate for the optimizer |
| `h_size` | 128 | Size of the hidden layers |

### PPO Hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| `gamma` | 0.99 | Discount factor |
| `lamda` | 0.95 | GAE parameter |
| `clip_param` | 0.2 | PPO clipping parameter |
| `buffer_size` | 256 | Number of samples to collect before updating the policy |
| `num_passes` | 2 | Number of update passes to make over the collected samples |
| `ent_coef` | 0.02 | Entropy coefficient |

### SAC Hyperparameters

| Parameter | Value | Description |
| --- | --- | --- |
| `gamma` | 0.99 | Discount factor |
| `tau` | 0.005 | Soft update coefficient |
| `alpha` | 0.2 | Entropy coefficient |
| `target_update_interval` | 2 | Number of steps between target network updates |
| `replay_buffer_size` | 1000000 | Size of the replay buffer |
| `warmup_steps` | 1000 | Number of steps to collect before training |
| `update_interval` | 4 | Number of steps between training updates |
