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


def test_plot_value_policy():
    env = GridEnv()
    agent = TDQ(env.state_size, env.action_space.n)
    plot_values_and_policy(agent, env, [9, 9], "Test Plot")


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


def test_td_q():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = TDQ(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_td_ac():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = TDAC(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_td_sr():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = TDSR(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_dyna_q():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = DynaQ(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_dyna_ac():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = DynaAC(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_dyna_sr():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = DynaSR(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_mbv():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = MBV(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_srmb():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = SRMB(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_qet():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = QET(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_distq():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = DistQ(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_qec():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = QEC(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_qmc():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = QMC(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_sarsa():
    env = GraphEnv(obs_type=GraphObsType.index)
    obs = env.reset()
    agent = SARSA(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_graph_episode():
    graph_env = GraphEnv()
    graph_agent = TDQ(graph_env.state_size, graph_env.action_space.n)
    _, _, _ = run_episode(graph_env, graph_agent, 100)


def test_grid_episode():
    grid_env = GridEnv()
    grid_agent = TDQ(grid_env.state_size, grid_env.action_space.n)
    _, _, _ = run_episode(grid_env, grid_agent, 100)
