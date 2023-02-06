import numpy as np
from neuronav.utils import onehot, twohot, run_episode, plot_values_and_policy
from neuronav.envs.graph_env import GraphEnv, GraphObsType
from neuronav.envs.grid_env import GridEnv, GridSize, GridObsType, OrientationType
from neuronav.envs.graph_structures import GraphStructure
from neuronav.envs.grid_topographies import GridTopography
from neuronav.agents.td_agents import QET, TDQ, TDAC, TDSR, SARSA
from neuronav.agents.dyna_agents import DynaQ, DynaAC, DynaSR
from neuronav.agents.mb_agents import MBV, SRMB
from neuronav.agents.mc_agents import QEC, QMC
from neuronav.agents.dist_agents import DistQ


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
    for obs_type in GraphObsType:
        env = GraphEnv(obs_type=obs_type)
        env.reset()
        env.step(env.action_space.sample())


def test_graph_structure():
    for structure in GraphStructure:
        env = GraphEnv(graph_structure=structure)
        env.reset()


def test_grid_orient():
    for obs_type in GridObsType:
        env = GridEnv(orientation_type=OrientationType.variable, obs_type=obs_type)
        obs = env.reset()
        env.step(env.action_space.sample())
        if obs_type != GridObsType.index:
            assert obs.shape == env.observation_space.shape


def test_grid_obs():
    for obs_type in GridObsType:
        env = GridEnv(obs_type=obs_type)
        obs = env.reset()
        env.step(env.action_space.sample())
        if obs_type != GridObsType.index:
            assert obs.shape == env.observation_space.shape


def test_grid_topo():
    for topo in GridTopography:
        for size in GridSize:
            env = GridEnv(topography=topo, grid_size=size)
            env.reset()


def test_seed_gridenv():
    env = GridEnv(seed=0, obs_type=GridObsType.symbolic)
    env.reset(stochasticity=1.0)
    obs_a, rew, don, _ = env.step(env.action_space.sample())
    env = GridEnv(seed=0, obs_type=GridObsType.symbolic)
    env.reset(stochasticity=1.0)
    obs_b, rew, don, _ = env.step(env.action_space.sample())
    assert obs_a.all() == obs_b.all()


def test_seed_graphenv():
    env = GraphEnv(seed=0, obs_type=GraphObsType.onehot)
    env.reset(stochasticity=1.0)
    obs_a, rew, don, _ = env.step(env.action_space.sample())
    env = GraphEnv(seed=0, obs_type=GraphObsType.onehot)
    env.reset(stochasticity=1.0)
    obs_b, rew, don, _ = env.step(env.action_space.sample())
    assert obs_a.all() == obs_b.all()
