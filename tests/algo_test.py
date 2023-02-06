from neuronav.utils import run_episode, plot_values_and_policy
from neuronav.envs.graph_env import GraphEnv
from neuronav.envs.grid_env import GridEnv, GridSize, GridObservation
from neuronav.agents.td_agents import QET, TDQ, TDAC, TDSR, SARSA
from neuronav.agents.dyna_agents import DynaQ, DynaAC, DynaSR
from neuronav.agents.mb_agents import MBV, SRMB
from neuronav.agents.mc_agents import QEC, QMC
from neuronav.agents.dist_agents import DistQ


def test_td_q():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = TDQ(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_td_ac():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = TDAC(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_td_sr():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = TDSR(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_dyna_q():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = DynaQ(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_dyna_ac():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = DynaAC(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_dyna_sr():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = DynaSR(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_mbv():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = MBV(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_srmb():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = SRMB(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_qet():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = QET(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_distq():
    env = GraphEnv(seed=0)
    obs = env.reset()
    agent = DistQ(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)


def test_qec():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = QEC(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_qmc():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = QMC(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_sarsa():
    env = GridEnv(obs_type=GridObservation.index, size=GridSize.micro)
    obs = env.reset()
    agent = SARSA(env.state_size, env.action_space.n)
    act = agent.sample_action(obs)
    env.step(act)
    for i in range(100):
        _, _, ep_return = run_episode(env, agent, 50)
    if ep_return == 1:
        assert True


def test_graph_episode():
    graph_env = GraphEnv()
    graph_agent = TDQ(graph_env.state_size, graph_env.action_space.n)
    _, _, _ = run_episode(graph_env, graph_agent, 100)


def test_grid_episode():
    grid_env = GridEnv()
    grid_agent = TDQ(grid_env.state_size, grid_env.action_space.n)
    _, _, _ = run_episode(grid_env, grid_agent, 100)


def test_plot_value_policy():
    env = GridEnv()
    agent = TDQ(env.state_size, env.action_space.n)
    plot_values_and_policy(agent, env, [9, 9], "Test Plot")
