from neuronav.deep_agents.ppo.agent import PPOAgent
from neuronav.deep_agents.sac.agent import SACAgent
from neuronav.envs.grid_env import GridEnv, GridObservation
from neuronav.envs.graph_env import GraphEnv, GraphObservation
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
    "grad_clip": 0.5,
}


visual64_types = [
    GridObservation.window,
    GridObservation.window_tight,
    GridObservation.visual,
    GridObservation.rendered_3d,
]

visual32_types = [
    GridObservation.images,
]


def get_model_params(env):
    if env.obs_mode in visual64_types:
        enc_type = "conv64"
    elif env.obs_mode in visual32_types:
        enc_type = "conv32"
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
        if obs_type != GridObservation.index and obs_type != GridObservation.ascii:
            env = GridEnv(obs_type=obs_type, torch_obs=True)
            agent = PPOAgent(get_model_params(env), agent_params_ppo)
            obs = env.reset()
            action = agent.sample_action(obs)
            _, _, _, _ = env.step(action)


def test_obs_types_sac():
    for obs_type in GridObservation:
        if obs_type != GridObservation.index and obs_type != GridObservation.ascii:
            env = GridEnv(obs_type=obs_type, torch_obs=True)
            agent = SACAgent(get_model_params(env), agent_params_sac)
            obs = env.reset()
            action = agent.sample_action(obs)
            _, _, _, _ = env.step(action)


def test_obs_types_graph_ppo():
    for obs_type in GraphObservation:
        if obs_type != GraphObservation.index:
            env = GraphEnv(obs_type=obs_type, torch_obs=True)
            agent = PPOAgent(get_model_params(env), agent_params_ppo)
            obs = env.reset()
            action = agent.sample_action(obs)
            _, _, _, _ = env.step(action)


def test_obs_types_graph_sac():
    for obs_type in GraphObservation:
        if obs_type != GraphObservation.index:
            env = GraphEnv(obs_type=obs_type, torch_obs=True)
            agent = SACAgent(get_model_params(env), agent_params_sac)
            obs = env.reset()
            action = agent.sample_action(obs)
            _, _, _, _ = env.step(action)


def test_ppo_update():
    env = GridEnv(obs_type=GridObservation.geometric)
    agent = PPOAgent(get_model_params(env), agent_params_ppo)
    for _ in range(10):
        obs = env.reset()
        for _ in range(100):
            obs = torch.Tensor(obs.copy())
            action = agent.sample_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.update([obs, action, next_obs, reward, done])
            obs = next_obs
            if done:
                break


def test_sac_update():
    env = GridEnv(obs_type=GridObservation.geometric)
    agent = SACAgent(get_model_params(env), agent_params_sac)
    for _ in range(10):
        obs = env.reset()
        for _ in range(100):
            obs = torch.Tensor(obs.copy())
            action = agent.sample_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.update([obs, action, next_obs, reward, done])
            obs = next_obs
            if done:
                break


def test_ppo_value():
    # call sample_value on an observation
    env = GridEnv(obs_type=GridObservation.geometric)
    agent = PPOAgent(get_model_params(env), agent_params_ppo)
    obs = env.reset()
    obs = torch.Tensor(obs.copy())
    value = agent.sample_value(obs)
    assert value.shape == (1,)
    assert value.dtype == torch.float32


def test_sac_value():
    # call sample_value on an observation
    env = GridEnv(obs_type=GridObservation.geometric)
    agent = SACAgent(get_model_params(env), agent_params_sac)
    obs = env.reset()
    obs = torch.Tensor(obs.copy())
    value = agent.sample_value(obs)
    assert value.shape == (1,)
    assert value.dtype == torch.float32
