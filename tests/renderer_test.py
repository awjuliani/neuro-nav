import unittest
from unittest.mock import MagicMock
from neuronav.envs.grid_3d import Grid3DRenderer


class TestGrid3DRenderer(unittest.TestCase):
    def setUp(self):
        self.renderer = Grid3DRenderer()
        self.mock_env = MagicMock()

    def test_set_camera(self):
        self.renderer.set_camera((1, 1), 0)
        self.renderer.set_camera((1, 1), 1)
        self.renderer.set_camera((1, 1), 2)
        self.renderer.set_camera((1, 1), 3)

        # No assertions here, just testing if the method runs without errors

    def test_render_walls(self):
        self.mock_env.blocks = [(1, 1), (2, 2)]
        self.mock_env.agent_pos = (0, 0)
        self.mock_env.looking = 1

        self.renderer.render_walls(self.mock_env.blocks)

        # No assertions here, just testing if the method runs without errors

    def test_render_objects(self):
        self.mock_env.objects = {
            "rewards": {(1, 1): 5, (2, 2): -5},
            "doors": [(3, 3)],
            "keys": [(4, 4)],
            "warps": [(5, 5)],
        }

        self.renderer.render_objects(self.mock_env)

        # No assertions here, just testing if the method runs without errors

    def tearDown(self):
        self.renderer.close()


if __name__ == "__main__":
    unittest.main()
