import time
import pytest
from neuronav.envs.grid_env import GridEnv, GridSize, GridObservation
from neuronav.envs.grid_templates import GridTemplate


def test_env_speed():
    env = GridEnv(
        template=GridTemplate.four_rooms_split,
        size=GridSize.small,
        obs_type=GridObservation.rendered_3d,
    )
    obs = env.reset()
    start_time = time.time()

    # take 1000 steps
    num_steps = 1000
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()

    # End timing
    end_time = time.time()

    # Calculate elapsed time and FPS
    elapsed_time = end_time - start_time
    fps = num_steps / elapsed_time

    env.close()

    # Assert that FPS is over 1000
    assert fps > 1000, f"FPS is {fps:.2f}, which is below the required 1000 FPS"

    print(f"Time taken for {num_steps} steps: {elapsed_time:.2f} seconds")
    print(f"Frames per second: {fps:.2f}")
