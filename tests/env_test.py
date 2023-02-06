import numpy as np
from neuronav.utils import onehot, twohot
from neuronav.envs.graph_env import GraphEnv, GraphObservation
from neuronav.envs.grid_env import GridEnv, GridSize, GridObservation, GridOrientation
from neuronav.envs.graph_templates import GraphTemplate
from neuronav.envs.grid_templates import GridTemplate


def test_one_hot():
    a = onehot(1, 5)
    assert a.all() == np.array([0, 1, 0, 0, 0]).all()


def test_two_hot():
    a = twohot([1, 1], 3)
    assert a.all() == np.array([0, 1, 0, 0, 1, 0]).all()


def test_objects_graph():
    objects = {"rewards": {0: 1}}
    env = GraphEnv()
    env.reset(objects=objects)
    env.step(env.action_space.sample())


def test_objects_grid():
    objects = {"rewards": {(1, 1): 1}, "markers": {(1, 1): (1, 0, 0)}}
    env = GridEnv()
    env.reset(objects=objects)
    env.step(env.action_space.sample())


def test_graph_obs():
    for obs_type in GraphObservation:
        env = GraphEnv(obs_type=obs_type)
        env.reset()
        env.step(env.action_space.sample())


def test_graph_templates():
    for template in GraphTemplate:
        env = GraphEnv(template=template)
        env.reset()


def test_grid_orient():
    for obs_type in GridObservation:
        env = GridEnv(orientation_type=GridOrientation.variable, obs_type=obs_type)
        obs = env.reset()
        env.step(env.action_space.sample())
        if obs_type != GridObservation.index:
            assert obs.shape == env.observation_space.shape


def test_grid_obs():
    for obs_type in GridObservation:
        env = GridEnv(obs_type=obs_type)
        obs = env.reset()
        env.step(env.action_space.sample())
        if obs_type != GridObservation.index:
            assert obs.shape == env.observation_space.shape


def test_grid_templates():
    for template in GridTemplate:
        for size in GridSize:
            env = GridEnv(template=template, size=size)
            env.reset()


def test_seed_gridenv():
    env = GridEnv(seed=0, obs_type=GridObservation.symbolic)
    env.reset(stochasticity=1.0)
    obs_a, rew, don, _ = env.step(env.action_space.sample())
    env = GridEnv(seed=0, obs_type=GridObservation.symbolic)
    env.reset(stochasticity=1.0)
    obs_b, rew, don, _ = env.step(env.action_space.sample())
    assert obs_a.all() == obs_b.all()


def test_seed_graphenv():
    env = GraphEnv(seed=0, obs_type=GraphObservation.onehot)
    env.reset(stochasticity=1.0)
    obs_a, rew, don, _ = env.step(env.action_space.sample())
    env = GraphEnv(seed=0, obs_type=GraphObservation.onehot)
    env.reset(stochasticity=1.0)
    obs_b, rew, don, _ = env.step(env.action_space.sample())
    assert obs_a.all() == obs_b.all()
