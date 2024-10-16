from ast import Dict
from gym import Env, spaces
import numpy as np
import neuronav.utils as utils
import random
import enum
from neuronav.envs.grid_templates import (
    generate_layout,
    GridTemplate,
    GridSize,
)
from neuronav.envs.grid_2d import Grid2DRenderer
import matplotlib.pyplot as plt
import cv2 as cv
import copy
import torch


class GridObservation(enum.Enum):
    onehot = "onehot"
    twohot = "twohot"
    geometric = "geometric"
    index = "index"
    boundary = "boundary"
    visual = "visual"
    images = "images"
    window = "window"
    symbolic = "symbolic"
    symbolic_window = "symbolic_window"
    window_tight = "window_tight"
    symbolic_window_tight = "symbolic_window_tight"
    rendered_3d = "rendered_3d"
    ascii = "ascii"


class GridOrientation(enum.Enum):
    fixed = "fixed"
    variable = "variable"


class GridEnv(Env):
    """
    Grid Environment. A 2D maze-like OpenAI gym compatible RL environment.

    Parameters
    ----------
    template : GridTemplate
        The layout template to use for the environment.
    size : GridSize
        The size of the grid (micro, small, large).
    obs_type : GridObservation
        The type of observation to use.
    orientation_type : GridOrientation
        The type of orientation to use.
    seed : int
        The seed to use for the environment.
    use_noop : bool
        Whether to include a no-op action in the action space.
    torch_obs : bool
        Whether to use torch observations.
        This converts the observation to a torch tensor.
        If the observation is an image, it will be in the shape (3, 64, 64).
    manual_collect : bool
        Whether to use the collect reward action (default == False).
    """

    def __init__(
        self,
        template: GridTemplate = GridTemplate.empty,
        size: GridSize = GridSize.small,
        obs_type: GridObservation = GridObservation.index,
        orientation_type: GridOrientation = GridOrientation.fixed,
        seed: int = None,
        use_noop: bool = False,
        torch_obs: bool = False,
        manual_collect: bool = False,
        resolution: int = 256,
    ):
        self.rng = np.random.RandomState(seed)
        self.resolution = resolution
        self.use_noop = use_noop
        self.manual_collect = manual_collect
        self.blocks, self.agent_start_pos, self.template_objects = generate_layout(
            template, size
        )
        self.grid_size = size.value
        self.renderer_2d = Grid2DRenderer(self.grid_size)
        self.state_size = self.grid_size * self.grid_size
        self.orientation_type = orientation_type
        self.torch_obs = torch_obs
        self.max_orient = 3
        self.set_action_space()
        self.agent_pos = [0, 0]
        self.base_objects = {
            "rewards": {},
            "markers": {},
            "keys": [],
            "doors": {},
            "warps": {},
        }
        self.direction_map = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])
        self.done = False
        self.keys = 0
        self.free_spots = self.make_free_spots()
        self.set_obs_space(obs_type)

    def set_action_space(self):
        if self.orientation_type == GridOrientation.variable:
            self.action_space = spaces.Discrete(3 + self.use_noop + self.manual_collect)
            self.orient_size = 4
        elif self.orientation_type == GridOrientation.fixed:
            self.orient_size = 1
            self.action_space = spaces.Discrete(4 + self.use_noop + self.manual_collect)
        else:
            raise Exception("No valid GridOrientation provided.")
        self.state_size *= self.orient_size

    def set_obs_space(self, obs_type):
        if isinstance(obs_type, str):
            obs_type = GridObservation(obs_type)
        self.obs_mode = obs_type
        if obs_type == GridObservation.visual:
            if self.torch_obs:
                self.obs_space = spaces.Box(0, 1, shape=(3, 64, 64))
            else:
                self.obs_space = spaces.Box(
                    0, 1, shape=(self.resolution, self.resolution, 3)
                )
        elif obs_type == GridObservation.onehot:
            self.obs_space = spaces.Box(
                0, 1, shape=(self.state_size * self.orient_size,), dtype=np.int32
            )
        elif obs_type == GridObservation.twohot:
            if self.orientation_type == GridOrientation.fixed:
                self.obs_space = spaces.Box(
                    0, 1, shape=(2 * self.grid_size,), dtype=np.int32
                )
            else:
                self.obs_space = spaces.Box(
                    0,
                    1,
                    shape=(2 * self.grid_size + self.orient_size,),
                    dtype=np.int32,
                )
        elif obs_type == GridObservation.geometric:
            if self.orientation_type == GridOrientation.fixed:
                self.obs_space = spaces.Box(0, 1, shape=(2,))
            else:
                self.obs_space = spaces.Box(0, 1, shape=(2 + self.orient_size,))
        elif obs_type == GridObservation.index:
            self.obs_space = spaces.Box(0, self.state_size, shape=(1,), dtype=np.int32)
        elif obs_type == GridObservation.boundary:
            self.ray_length = self.grid_size
            self.num_rays = 4
            if self.orientation_type == GridOrientation.fixed:
                self.obs_space = spaces.Box(0, 1, shape=(self.num_rays,))
            else:
                self.obs_space = spaces.Box(
                    0, 1, shape=(self.num_rays + self.orient_size,)
                )
        elif obs_type == GridObservation.images:
            self.obs_space = spaces.Box(0, 1, shape=(32, 32, 3))
            self.images = utils.cifar10()[0]
        elif obs_type == GridObservation.window:
            if self.torch_obs:
                self.obs_space = spaces.Box(0, 1, shape=(3, 64, 64))
            else:
                self.obs_space = spaces.Box(
                    0, 1, shape=(self.resolution, self.resolution, 3)
                )
        elif obs_type == GridObservation.window_tight:
            if self.torch_obs:
                self.obs_space = spaces.Box(0, 1, shape=(3, 64, 64))
            else:
                self.obs_space = spaces.Box(
                    0, 1, shape=(self.resolution, self.resolution, 3)
                )
        elif obs_type == GridObservation.symbolic:
            self.obs_space = spaces.Box(0, 1, shape=(self.grid_size, self.grid_size, 6))
        elif obs_type == GridObservation.symbolic_window:
            self.obs_space = spaces.Box(0, 1, shape=(5, 5, 6))
        elif obs_type == GridObservation.symbolic_window_tight:
            self.obs_space = spaces.Box(0, 1, shape=(3, 3, 6))
        elif obs_type == GridObservation.rendered_3d:
            if self.torch_obs:
                self.obs_space = spaces.Box(0, 1, shape=(3, 64, 64))
            else:
                self.obs_space = spaces.Box(
                    0, 1, shape=(self.resolution, self.resolution, 3)
                )
            from neuronav.envs.grid_3d import Grid3DRenderer

            self.renderer_3d = Grid3DRenderer(self.resolution)
        elif obs_type == GridObservation.ascii:
            self.obs_space = spaces.Box(0, 1, shape=(self.grid_size, self.grid_size))
        else:
            raise Exception("No valid ObservationType provided.")

    def reset(
        self,
        objects: Dict = None,
        agent_pos: list = None,
        episode_length: int = 100,
        random_start: bool = False,
        terminate_on_reward: bool = True,
        time_penalty: float = 0.0,
        stochasticity: float = 0.0,
        visible_walls: bool = True,
    ):
        """
        Resets the environment to its initial configuration.
        Args:
            objects: A dictionary of objects to be placed in the environment.
            agent_pos: The optional starting position of the agent.
            episode_length: The maximum number of steps in an episode.
            random_start: Whether to start the agent at a random position.
            terminate_on_reward: Whether to terminate the episode when the agent
                receives a reward.
            time_penalty: The reward penalty for each step taken in the environment.
            stochasticity: The probability of the agent taking a random action.
            visible_walls: Whether the agent can see the walls of the environment.
        Returns:
            The initial observation of the environment.
        """
        self.done = False
        self.episode_time = 0
        self.orientation = 0
        self.looking = 0
        self.keys = 0
        self.time_penalty = time_penalty
        self.max_episode_time = episode_length
        self.terminate_on_reward = terminate_on_reward
        self.stochasticity = stochasticity
        self.visible_walls = visible_walls
        self.cached_objects = None

        if agent_pos is not None:
            self.agent_pos = agent_pos
        elif random_start:
            self.agent_pos = self.get_free_spot()
        else:
            self.agent_pos = self.agent_start_pos

        base_object = copy.deepcopy(self.base_objects)
        if objects is not None:
            use_objects = copy.deepcopy(objects)
        else:
            use_objects = copy.deepcopy(self.template_objects)
        for key in use_objects.keys():
            if key in base_object.keys():
                base_object[key] = use_objects[key]
        self.objects = base_object
        return self.observation

    def get_free_spot(self):
        return random.choice(self.free_spots)

    def make_free_spots(self):
        return [
            [i, j]
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if [i, j] not in self.blocks
        ]

    def make_symbolic_obs(self):
        """
        Returns a symbolic representation of the environment in a numpy tensor.
        Tensor shape is (grid_size, grid_size, 6)
        6 channels are:
            0: agent
            1: rewards
            2: keys
            3: doors
            4: walls
            5: warps
        """
        grid = np.zeros([self.grid_size, self.grid_size, 6])

        # Set agent's position
        grid[self.agent_pos[0], self.agent_pos[1], 0] = 1

        # Set rewards
        reward_list = [
            (loc, reward)
            for loc, reward in self.objects["rewards"].items()
            if type(reward) != list or reward[1] == 1
        ]
        for loc, reward in reward_list:
            if type(reward) == list:
                reward = reward[0]
            grid[loc[0], loc[1], 1] = reward

        # Set keys
        key_locs = self.objects["keys"]
        for loc in key_locs:
            grid[loc[0], loc[1], 2] = 1

        # Set doors
        door_locs = self.objects["doors"]
        for loc in door_locs:
            grid[loc[0], loc[1], 3] = 1

        # Set warps
        warp_locs = self.objects["warps"].keys()
        for loc in warp_locs:
            grid[loc[0], loc[1], 5] = 1

        # Set walls
        if self.visible_walls:
            walls = self.render_walls()
            grid[:, :, 4] = walls

        return grid

    def render_walls(self):
        """
        Returns a numpy array of the walls in the environment.
        """
        grid = np.zeros([self.grid_size, self.grid_size])
        for block in self.blocks:
            grid[block[0], block[1]] = 1
        return grid

    def make_symbolic_window_obs(self, size: int = 5):
        if size not in [3, 5]:
            raise ValueError("Window size must be 3 or 5")

        obs = self.make_symbolic_obs()
        pad_size = (size - 1) // 2
        full_window = np.pad(
            obs,
            ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        full_window[:, :, 4] = np.where(
            full_window[:, :, 4] == 0, 1, full_window[:, :, 4]
        )

        window = full_window[
            self.agent_pos[0] : self.agent_pos[0] + size,
            self.agent_pos[1] : self.agent_pos[1] + size,
            :,
        ]

        return window

    def render(self, provide=False, mode="human"):
        image = self.renderer_2d.render_frame(self)
        if self.obs_mode == GridObservation.rendered_3d:
            img_3d = self.renderer_3d.render_frame(self)
            img_2d_resized = cv.resize(
                image,
                (self.resolution, self.resolution),
                interpolation=cv.INTER_NEAREST,
            )
            image = np.concatenate((img_3d, img_2d_resized), axis=1)

        if mode == "human":
            plt.imshow(image)
            plt.axis("off")
            plt.show()
        if provide:
            return image

    def make_visual_obs(self):
        img = self.renderer_2d.render_frame(self)
        return self.resize_obs(img)

    def make_window_obs(self, w_size=2):
        img = self.renderer_2d.render_window(self, w_size)
        return self.resize_obs(img)

    def make_3d_obs(self):
        img = self.renderer_3d.render_frame(self)
        return self.resize_obs(img)

    def resize_obs(self, img):
        if self.torch_obs:
            img = cv.resize(img, (64, 64), interpolation=cv.INTER_NEAREST)
            img = np.moveaxis(img, 2, 0) / 255.0
        else:
            img = cv.resize(
                img, (self.resolution, self.resolution), interpolation=cv.INTER_NEAREST
            )
        return img

    def move_agent(self, direction: np.array):
        """
        Moves the agent in the given direction.
        """
        new_pos = self.agent_pos + direction
        if self.check_target(new_pos):
            self.agent_pos = list(new_pos)

    def check_target(self, target: list):
        """
        Checks if the target is a valid (movable) position.
        Returns True if the target is valid, False otherwise.
        """
        target_tuple = tuple(target)
        target_list = list(target_tuple)
        x_check = -1 < target[0] < self.grid_size
        y_check = -1 < target[1] < self.grid_size

        if not (x_check and y_check):
            return False

        if target_list in self.blocks:
            return False

        if target_tuple in self.objects["doors"]:
            if self.keys > 0:
                self.objects["doors"].pop(target_tuple)
                self.keys -= 1
            else:
                return False

        return True

    def get_observation(self, perspective: list):
        """
        Returns an observation corresponding to the provided coordinates.
        """
        if self.obs_mode == GridObservation.onehot:
            return self.make_onehot_obs(perspective)
        elif self.obs_mode == GridObservation.twohot:
            return self.make_twohot_obs(perspective)
        elif self.obs_mode == GridObservation.geometric:
            return self.make_geometric_obs(perspective)
        elif self.obs_mode == GridObservation.visual:
            return self.make_visual_obs()
        elif self.obs_mode == GridObservation.index:
            return self.make_index_obs(perspective)
        elif self.obs_mode == GridObservation.boundary:
            return self.make_boundary_obs(perspective)
        elif self.obs_mode == GridObservation.images:
            return self.make_image_obs(perspective)
        elif self.obs_mode == GridObservation.window:
            return self.make_window_obs()
        elif self.obs_mode == GridObservation.symbolic:
            return self.make_symbolic_obs()
        elif self.obs_mode == GridObservation.symbolic_window:
            return self.make_symbolic_window_obs()
        elif self.obs_mode == GridObservation.window_tight:
            return self.make_window_obs(w_size=1)
        elif self.obs_mode == GridObservation.symbolic_window_tight:
            return self.make_symbolic_window_obs(size=3)
        elif self.obs_mode == GridObservation.rendered_3d:
            return self.make_3d_obs()
        elif self.obs_mode == GridObservation.ascii:
            return self.make_ascii_obs()
        else:
            raise ValueError("Invalid observation mode.")

    def make_onehot_obs(self, perspective: list):
        # one-hot encoding of the perspective
        one_hot = utils.onehot(
            self.orientation * self.grid_size * self.grid_size
            + perspective[0] * self.grid_size
            + perspective[1],
            self.state_size * self.orient_size,
        )
        return one_hot

    def make_twohot_obs(self, perspective: list):
        two_hot = utils.twohot(perspective, self.grid_size)
        if self.orientation_type == GridOrientation.variable:
            two_hot = np.concatenate(
                [two_hot, utils.onehot(self.orientation, self.orient_size)]
            )
        return two_hot

    def make_geometric_obs(self, perspective: list):
        geo = np.array(perspective) / (self.grid_size - 1.0)
        if self.orientation_type == GridOrientation.variable:
            geo = np.concatenate([geo, utils.onehot(self.orientation, self.orient_size)])
        return geo

    def make_index_obs(self, perspective: list):
        idx = (
            self.orientation * self.grid_size * self.grid_size
            + perspective[0] * self.grid_size
            + perspective[1]
        )
        return idx

    def make_boundary_obs(self, perspective: list):
        bounds = self.get_boundaries(
            perspective, False, self.num_rays, self.ray_length
        )
        if self.orientation_type == GridOrientation.variable:
            bounds = np.concatenate(
                [bounds, utils.onehot(self.orientation, self.orient_size)]
            )
        return bounds

    def make_image_obs(self, perspective: list):
        idx = (
            self.orientation * self.state_size
            + perspective[0] * self.grid_size
            + perspective[1]
        )
        return np.rot90(self.images[idx], k=3)

    def make_ascii_obs(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        for block in self.blocks:
            grid[block[0], block[1]] = 2
        for reward_pos, reward_val in self.objects["rewards"].items():
            if reward_val > 0:
                grid[reward_pos[0], reward_pos[1]] = 3
            else:
                grid[reward_pos[0], reward_pos[1]] = 4
        for key_pos in self.objects["keys"]:
            grid[key_pos[0], key_pos[1]] = 5
        for door_pos in self.objects["doors"]:
            grid[door_pos[0], door_pos[1]] = 6
        for warp_pos in self.objects["warps"]:
            grid[warp_pos[0], warp_pos[1]] = 7
        # _ = empty, A = agent, B = block, R = reward, L = lava, K = key, D = door, W = warp
        ascii_map = {0: " ", 1: "A", 2: "B", 3: "R", 4: "L", 5: "K", 6: "D", 7: "W"}
        ascii_grid = np.vectorize(ascii_map.get)(grid)
        # join with newlines
        ascii_str = "\n".join(["".join(row) for row in ascii_grid])
        return ascii_str

    @property
    def observation(self):
        if self.torch_obs:
            return torch.Tensor(self.get_observation(self.agent_pos).copy())
        return self.get_observation(self.agent_pos)

    def get_boundaries(
        self,
        object_point: list,
        use_onehot: bool = False,
        num_rays: int = 4,
        ray_length: int = 10,
    ):
        def normalize_distance(distance: int):
            if use_onehot:
                return utils.onehot(distance, ray_length)
            return distance / self.grid_size

        if num_rays == 4:
            ray_angles = [0, 2, 4, 6]
        else:
            ray_angles = [6, 0, 2]

        distances = [
            normalize_distance(self.simple_ray(angle, object_point))
            for angle in ray_angles
        ]

        return np.stack(distances).reshape(-1)

    def simple_ray(self, direction: int, start: list):
        """
        Returns the distance to the nearest object in the given direction.
        """
        if self.orientation_type == GridOrientation.variable:
            direction = (direction + self.orientation * 2) % 8

        ray_length = self.grid_size

        count = 0
        hit = False
        try_pos = start.copy()

        moves = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        while not hit:
            dx, dy = moves[direction]
            try_pos = [try_pos[0] + dx, try_pos[1] + dy]
            hit = not self.check_target(try_pos) or count == ray_length
            count += 1

        return count - 1

    def rotate(self, direction: int):
        """
        Rotates the agent orientation in the given direction.
        """
        self.orientation = (self.orientation + direction) % (self.max_orient + 1)

    def step(self, action: int):
        """
        Steps the environment forward given an action.
        Action is an integer in the range [0, self.action_space.n).
        """
        if self.done:
            print("Episode finished. Please reset the environment.")
            return None, None, None, None

        if self.stochasticity > self.rng.rand():
            action = self.rng.randint(0, self.action_space.n)

        if self.manual_collect == True:
            can_collect = False
        else:
            can_collect = True

        if self.orientation_type == GridOrientation.variable:
            # 0 - Counter-clockwise rotation
            # 1 - Clockwise rotation
            # 2 - Forward movement
            # 3 - Stay (if flag use_noop == True)/ Collect (if flag manual_collect == True)
            # 4 - Collect (if flag manual_collect == True)
            if action == 0:
                self.rotate(-1)
            elif action == 1:
                self.rotate(1)
            elif action == 2:
                move_array = self.direction_map[self.orientation]
                self.move_agent(move_array)
            elif action == 3:
                if self.use_noop:
                    pass
                else:
                    can_collect = True
            elif action == 4:
                can_collect = True

            self.looking = self.orientation
        else:
            # 0 - North
            # 1 - East
            # 2 - South
            # 3 - West
            # 4 - Stay (if flag use_noop == True) / Collect (if flag manual_collect == True)
            # 5 - Collect (if flag manual_collect == True)
            move_array = self.direction_map[action]
            if action < 4:
                self.looking = action
            elif action == 4:
                if self.use_noop:
                    pass
                else:
                    can_collect = True
            elif action == 5:
                can_collect = True

            self.move_agent(move_array)

        self.episode_time += 1
        reward = 0 if action == 4 else self.time_penalty
        eval_pos = tuple(self.agent_pos)
        terminate = self.terminate_on_reward

        if (eval_pos in self.objects["rewards"]) & can_collect == True:
            reward_info = self.objects["rewards"][eval_pos]
            if isinstance(reward_info, list):
                terminate = reward_info[2]
                reward_val = reward_info[0]
            else:
                reward_val = reward_info
            reward += reward_val
            if terminate:
                self.done = True
            self.objects["rewards"].pop(eval_pos)

        if eval_pos in self.objects["keys"]:
            self.keys += 1
            self.objects["keys"].remove(eval_pos)

        if eval_pos in self.objects["warps"]:
            self.agent_pos = self.objects["warps"][eval_pos]

        return self.observation, reward, self.done, {}

    def close(self) -> None:
        if self.obs_mode == GridObservation.rendered_3d:
            self.renderer_3d.close()
        return super().close()
