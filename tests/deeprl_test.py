from neuronav.deep_agents.ppo.agent import PPOAgent
from neuronav.deep_agents.ppo.model import PPOModel
from neuronav.deep_agents.sac.agent import SACAgent
from neuronav.deep_agents.sac.model import SACModel
from neuronav.envs.grid_env import GridEnv, GridObservation
import math
import torch


agent_params_sac = {
    "batch_size": 32,
    "gamma": 0.99,
    "tau": 0.005,
    "target_update_interval": 2,
    "replay_buffer_size": 100000,
    "update_interval": 4,
    "warmup_steps": 1000,
    "alpha": 0.02,
}

agent_params_ppo = {
    "gamma": 0.99,
    "lambda": 0.95,
    "buffer_size": 256,
    "ent_coef": 0.02,
    "num_passes": 2,
    "clip_param": 0.2,
    "batch_size": 32,
}


def get_model_params(env):
    if env.obs_mode == GridObservation.window:
        enc_type = "conv64"
    else:
        enc_type = "linear"
    model_params = {
        "enc_type": enc_type,
        "h_size": 128,
        "lr": 3e-4,
        "obs_size": math.prod(env.obs_space.shape),
        "act_size": env.action_space.n,
        "depth": 3,
    }
    return model_params


# Test all observation types
def test_obs_types_ppo():
    for obs_type in GridObservation:
        if obs_type != GridObservation.index:
            print(f"Testing obs_type: {obs_type}")
            env = GridEnv(obs_type=obs_type)
            agent = PPOAgent(get_model_params(env), agent_params_ppo)
            obs = env.reset()
            print(obs.shape)
            obs = torch.Tensor(obs.copy())
            action = agent.sample_action(obs)
            _, _, _, _ = env.step(action)


def test_obs_types_sac():
    for obs_type in GridObservation:
        if obs_type != GridObservation.index:
            env = GridEnv(obs_type=obs_type)
            agent = SACAgent(get_model_params(env), agent_params_sac)
            obs = env.reset()
            obs = torch.Tensor(obs.copy())
            action = agent.sample_action(obs)
            _, _, _, _ = env.step(action)
