from neuronav.envs.graph_env import GraphEnv, GraphObservation
from neuronav.envs.graph_templates import GraphTemplate
import pytest


def test_objects_graph():
    objects = {"rewards": {0: 1}}
    env = GraphEnv()
    env.reset(objects=objects)
    env.step(env.action_space.sample())


def test_graph_obs():
    for obs_type in GraphObservation:
        env = GraphEnv(obs_type=obs_type)
        obs = env.reset()
        env.step(env.action_space.sample())
        if obs_type != GraphObservation.index:
            assert obs.shape == env.obs_space.shape


def test_graph_templates():
    for template in GraphTemplate:
        env = GraphEnv(template=template)
        env.reset()


def test_seed_graphenv():
    env = GraphEnv(seed=0, obs_type=GraphObservation.onehot)
    env.reset(stochasticity=1.0)
    obs_a, rew, don, _ = env.step(env.action_space.sample())
    env = GraphEnv(seed=0, obs_type=GraphObservation.onehot)
    env.reset(stochasticity=1.0)
    obs_b, rew, don, _ = env.step(env.action_space.sample())
    assert obs_a.all() == obs_b.all()


def test_graph_reset():
    env = GraphEnv()
    env.reset(agent_pos=2)
    assert env.agent_pos == 2

    env.reset(random_start=True)
    assert env.agent_pos >= 0 and env.agent_pos < env.state_size

    env.reset(time_penalty=0.1)
    assert env.time_penalty == 0.1

    env.reset(stochasticity=0.5)
    assert env.stochasticity == 0.5

    env.reset(objects={"rewards": {1: 1}})
    assert env.objects["rewards"][1] == 1


def test_step_noop():
    env = GraphEnv(use_noop=True)
    env.reset()
    agent_pos_before = env.agent_pos
    obs, reward, done, _ = env.step(env.action_space.n - 1)
    assert agent_pos_before == env.agent_pos


def test_step_reward():
    env = GraphEnv()
    env.reset(objects={"rewards": {1: 1}})
    env.agent_pos = 0
    obs, reward, done, _ = env.step(0)
    assert reward == 1


def test_render():
    env = GraphEnv()
    env.reset()
    try:
        env.render()
    except Exception as e:
        pytest.fail(f"Render failed with exception: {e}")
