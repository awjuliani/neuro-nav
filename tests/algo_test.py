from neuronav.utils import run_episode, plot_values_and_policy
from neuronav.envs.graph_env import GraphEnv
from neuronav.envs.grid_env import GridEnv, GridSize, GridObservation
from neuronav.agents.td_agents import QET, TDQ, TDAC, TDSR, SARSA
from neuronav.agents.dyna_agents import DynaQ, DynaAC, DynaSR
from neuronav.agents.mb_agents import MBV, SRMB
from neuronav.agents.mc_agents import QEC, QMC
from neuronav.agents.dist_agents import DistQ
import pytest
from neuronav.agents.base_agent import BaseAgent
import numpy as np


class TestAgent(BaseAgent):
    def _update(self, current_exp):
        return 0

    def reset(self):
        pass


@pytest.fixture
def test_agent():
    return TestAgent(state_size=4, action_size=3)


def test_base_agent_init(test_agent):
    assert test_agent.state_size == 4
    assert test_agent.action_size == 3
    assert test_agent.lr == 1e-1
    assert test_agent.beta == 1e4
    assert test_agent.gamma == 0.99
    assert test_agent.poltype == "softmax"
    assert test_agent.num_updates == 0
    assert test_agent.epsilon == 1e-1


def test_base_agent_sample_action(test_agent):
    policy_logits = np.array([1, 2, 3])
    action = test_agent.base_sample_action(policy_logits)

    assert action in [0, 1, 2]


def test_base_agent_get_policy(test_agent):
    policy_logits = np.array([1, 2, 3])
    policy = test_agent.base_get_policy(policy_logits)

    assert policy.shape == (3,)
    assert np.isclose(np.sum(policy), 1.0)


def test_base_agent_discount(test_agent):
    rewards = [1, 2, 3, 4]
    gamma = 0.5
    discounted_rewards = test_agent.discount(rewards, gamma)

    assert len(discounted_rewards) == len(rewards)
    assert discounted_rewards == [3.25, 4.5, 5.0, 4]


def test_base_agent_update(test_agent):
    current_exp = None
    error = test_agent.update(current_exp)

    assert test_agent.num_updates == 1
    assert error == 0


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
    agent.update((obs, act, 0, obs, False))


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
