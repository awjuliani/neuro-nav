from ast import Dict
from gym import Env, spaces
import numpy as np
import neuronav.utils as utils
import random
import enum
from neuronav.envs.grid_topographies import (
    generate_topography,
    GridTopography,
    GridSize,
)
import matplotlib.pyplot as plt
import cv2 as cv


class GridObsType(enum.Enum):
    onehot = "onehot"
    twohot = "twohot"
    geometric = "geometric"
    index = "index"
    boundary = "boundary"
    visual = "visual"
    images = "images"
    window = "window"


class OrientationType(enum.Enum):
    fixed = "fixed"
    variable = "variable"


class GridEnv(Env):
    """
    Grid Environment. A 2D maze-like OpenAI gym compatible RL environment.
    """

    def __init__(
        self,
        topography: GridTopography = GridTopography.empty,
        grid_size: GridSize = GridSize.small,
        obs_type: GridObsType = GridObsType.index,
        orientation_type: OrientationType = OrientationType.fixed,
    ):
        self.blocks, self.agent_start_pos, self.topo_reward_locs = generate_topography(
            topography, grid_size
        )
        self.grid_size = grid_size.value
        self.state_size = self.grid_size * self.grid_size
        self.orientation_type = orientation_type
        self.max_orient = 3
        if self.orientation_type == OrientationType.variable:
            self.action_space = spaces.Discrete(3)
            self.orient_size = 4
        elif self.orientation_type == OrientationType.fixed:
            self.orient_size = 1
            self.action_space = spaces.Discrete(4)
        else:
            raise Exception("No valid OrientationType provided.")
        self.state_size *= self.orient_size
        self.agent_pos = [0, 0]
        self.reward_locs = {}
        self.direction_map = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])
        self.done = False
        self.free_spots = self.make_free_spots()
        if isinstance(obs_type, str):
            obs_type = GridObsType(obs_type)
        self.obs_mode = obs_type
        if obs_type == GridObsType.visual:
            self.observation_space = spaces.Box(
                0,
                1,
                shape=(
                    110,
                    110,
                    3,
                ),
            )
        elif obs_type == GridObsType.onehot:
            self.observation_space = spaces.Box(
                0, 1, shape=(self.state_size * self.orient_size,), dtype=np.int32
            )
        elif obs_type == GridObsType.twohot:
            if self.orientation_type == OrientationType.fixed:
                self.observation_space = spaces.Box(
                    0, 1, shape=(2 * self.grid_size,), dtype=np.int32
                )
            else:
                self.observation_space = spaces.Box(
                    0,
                    1,
                    shape=(2 * self.grid_size + self.orient_size,),
                    dtype=np.int32,
                )
        elif obs_type == GridObsType.geometric:
            if self.orientation_type == OrientationType.fixed:
                self.observation_space = spaces.Box(0, 1, shape=(2,))
            else:
                self.observation_space = spaces.Box(0, 1, shape=(2 + self.orient_size,))
        elif obs_type == GridObsType.index:
            self.observation_space = spaces.Box(
                0, self.state_size, shape=(1,), dtype=np.int32
            )
        elif obs_type == GridObsType.boundary:
            self.ray_length = self.grid_size
            self.num_rays = 4
            if self.orientation_type == OrientationType.fixed:
                self.observation_space = spaces.Box(
                    0,
                    1,
                    shape=(self.num_rays,),
                )
            else:
                self.observation_space = spaces.Box(
                    0,
                    1,
                    shape=(self.num_rays + self.orient_size,),
                )
        elif obs_type == GridObsType.images:
            self.observation_space = spaces.Box(0, 1, shape=(32, 32, 3))
            self.images, _, _, _ = utils.cifar10()
        elif obs_type == GridObsType.window:
            self.observation_space = spaces.Box(
                0,
                1,
                shape=(
                    64,
                    64,
                    3,
                ),
            )
        else:
            raise Exception("No valid ObservationType provided.")

    def reset(
        self,
        reward_locs: Dict = None,
        agent_pos: list = None,
        episode_length: int = 100,
        random_start: bool = False,
        terminate_on_reward: bool = True,
        time_penalty: float = 0.0,
    ):
        """
        Resets the environment to its initial configuration.
        """
        self.done = False
        self.episode_time = 0
        self.orientation = 0
        self.looking = 0
        self.time_penalty = time_penalty
        self.max_episode_time = episode_length
        self.terminate_on_reward = terminate_on_reward

        if agent_pos != None:
            self.agent_pos = agent_pos
        elif random_start:
            self.agent_pos = self.get_free_spot()
        else:
            self.agent_pos = self.agent_start_pos

        if reward_locs != None:
            self.reward_locs = reward_locs
        else:
            self.reward_locs = self.topo_reward_locs
        return self.observation

    def get_free_spot(self):
        selection = random.choice(self.free_spots)
        return selection

    def make_free_spots(self):
        free_spots = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if [i, j] not in self.blocks:
                    free_spots.append([i, j])
        return free_spots

    def grid(self, render_objects: bool = True):
        grid = np.zeros([self.grid_size, self.grid_size, 3])
        if render_objects:
            grid[self.agent_pos[0], self.agent_pos[1], :] = 1
            for loc, reward in self.reward_locs.items():
                if reward > 0:
                    grid[loc[0], loc[1], 1] = np.clip(np.sqrt(reward), 0, 1)
                else:
                    grid[loc[0], loc[1], 0] = np.clip(np.sqrt(np.abs(reward)), 0, 1)
        for block in self.blocks:
            grid[block[0], block[1], :] = 0.5
        return grid

    def render(self):
        image = self.make_visual_obs()
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    def get_square_edges(self, x, y, unit_size, block_size):
        # swap x and y because of the way cv2 draws images
        x, y = y, x
        block_border = block_size // 10
        true_start = unit_size - block_size + 1
        block_end = block_size - block_border * 2 + 1
        return (
            (x * unit_size + true_start, y * unit_size + true_start),
            (x * unit_size + block_end, y * unit_size + block_end),
        )

    def make_visual_obs(self, resize=False):
        block_size = 20
        img_size = block_size * self.grid_size
        img = np.ones((img_size, img_size, 3), np.uint8) * 225
        block_border = block_size // 10

        # draw thin lines to separate each position
        for i in range(self.grid_size + 1):
            cv.line(
                img, (0, i * block_size), (img_size, i * block_size), (210, 210, 210), 1
            )
            cv.line(
                img, (i * block_size, 0), (i * block_size, img_size), (210, 210, 210), 1
            )
        # draw the blocks
        for x, y in self.blocks:
            start, end = self.get_square_edges(x, y, block_size, block_size - 2)
            cv.rectangle(img, start, end, (150, 150, 150), -1)
            cv.rectangle(img, start, end, (100, 100, 100), block_border - 1)
        # draw the reward locations
        for pos, reward in self.reward_locs.items():
            if reward > 0:
                fill_color = (100, 100, 255)
                border_color = (50, 50, 200)
            else:
                fill_color = (255, 100, 100)
                border_color = (200, 50, 50)
            start, end = self.get_square_edges(
                pos[0], pos[1], block_size, block_size - 4
            )
            cv.rectangle(img, start, end, fill_color, -1)
            cv.rectangle(img, start, end, border_color, block_border - 1)
        # draw the agent as an isosoceles triangle
        agent_pos = self.agent_pos
        agent_dir = self.looking
        agent_color = (0, 0, 0)
        agent_size = block_size // 2
        agent_offset = block_size // 4
        # swap x and y because of the way cv2 draws images
        agent_pos = (agent_pos[1], agent_pos[0])
        x_offset = agent_pos[0] * block_size + agent_offset
        y_offset = agent_pos[1] * block_size + agent_offset
        if agent_dir == 2:
            # facing down
            pts = np.array(
                [
                    (x_offset, y_offset),
                    (x_offset + agent_size, y_offset),
                    (x_offset + agent_size // 2, y_offset + agent_size),
                ]
            )
        elif agent_dir == 3:
            # facing left
            pts = np.array(
                [
                    (x_offset + agent_size, y_offset),
                    (x_offset + agent_size, y_offset + agent_size),
                    (x_offset, y_offset + agent_size // 2),
                ]
            )
        elif agent_dir == 0:
            # facing up
            pts = np.array(
                [
                    (x_offset, y_offset + agent_size),
                    (x_offset + agent_size, y_offset + agent_size),
                    (x_offset + agent_size // 2, y_offset),
                ]
            )
        elif agent_dir == 1:
            # facing right
            pts = np.array(
                [
                    (x_offset, y_offset),
                    (x_offset, y_offset + agent_size),
                    (x_offset + agent_size, y_offset + agent_size // 2),
                ]
            )
        cv.fillConvexPoly(img, pts, agent_color)
        if resize:
            img = cv.resize(img, (110, 110))
        return img

    def make_window(self, w_size=2, block_size=20, resize=True):
        base_image = self.make_visual_obs()
        template_size = block_size * (self.grid_size + 2)
        template = np.ones((template_size, template_size, 3), dtype=np.int8) * 150
        template[block_size:-block_size, block_size:-block_size, :] = base_image
        x, y = self.agent_pos
        start_x = block_size * (x - w_size + 1)
        end_x = block_size * (x + w_size + 2)
        start_y = block_size * (y - w_size + 1)
        end_y = block_size * (y + w_size + 2)
        window = template[start_x:end_x, start_y:end_y]
        if resize:
            window = cv.resize(window, (64, 64))
        return window

    def move_agent(self, direction: np.array):
        new_pos = self.agent_pos + direction
        if self.check_target(new_pos):
            self.agent_pos = list(new_pos)

    def check_target(self, target: list):
        x_check = -1 < target[0] < self.grid_size
        y_check = -1 < target[1] < self.grid_size
        block_check = list(target) not in self.blocks
        return x_check and y_check and block_check

    def get_observation(self, perspective: list):
        """
        Returns an observation corresponding to the provided coordinates.
        """
        if self.obs_mode == GridObsType.onehot:
            one_hot = utils.onehot(
                self.orientation * self.grid_size * self.grid_size
                + perspective[0] * self.grid_size
                + perspective[1],
                self.state_size * self.orient_size,
            )
            return one_hot
        elif self.obs_mode == GridObsType.twohot:
            two_hot = utils.twohot(perspective, self.grid_size)
            if self.orientation_type == OrientationType.variable:
                two_hot = np.concatenate(
                    [two_hot, utils.onehot(self.orientation, self.orient_size)]
                )
            return two_hot
        elif self.obs_mode == GridObsType.geometric:
            geo = np.array(perspective) / (self.grid_size - 1.0)
            if self.orientation_type == OrientationType.variable:
                geo = np.concatenate(
                    [geo, utils.onehot(self.orientation, self.orient_size)]
                )
            return geo
        elif self.obs_mode == GridObsType.visual:
            return self.make_visual_obs(True)
        elif self.obs_mode == GridObsType.index:
            idx = (
                self.orientation * self.grid_size * self.grid_size
                + perspective[0] * self.grid_size
                + perspective[1]
            )
            return idx
        elif self.obs_mode == GridObsType.boundary:
            bounds = self.get_boundaries(
                perspective, False, self.num_rays, self.ray_length
            )
            if self.orientation_type == OrientationType.variable:
                bounds = np.concatenate(
                    [bounds, utils.onehot(self.orientation, self.orient_size)]
                )
            return bounds
        elif self.obs_mode == GridObsType.images:
            idx = (
                self.orientation * self.state_size
                + perspective[0] * self.grid_size
                + perspective[1]
            )
            return np.rot90(self.images[idx], k=3)
        elif self.obs_mode == GridObsType.window:
            return self.make_window()

    @property
    def observation(self):
        return self.get_observation(self.agent_pos)

    def get_boundaries(
        self,
        object_point: list,
        use_onehot: bool = False,
        num_rays: int = 4,
        ray_length: int = 10,
    ):
        distances = []
        if num_rays == 4:
            nums = [0, 2, 4, 6]
        else:
            nums = [6, 0, 2]
        for num in nums:
            distance = self.simple_ray(num, object_point)
            if use_onehot:
                distance = utils.onehot(distance, ray_length)
            else:
                distance = distance / self.grid_size
            distances.append(distance)
        distances = np.stack(distances)
        return distances.reshape(-1)

    def simple_ray(self, direction: int, start: list):
        if self.orientation_type == OrientationType.variable:
            direction += self.orientation * 2
            if direction > 7:
                direction -= 8

        ray_length = self.grid_size

        count = -1
        hit = False
        try_pos = start

        while not hit:
            count += 1
            if direction == 0:
                try_pos = [try_pos[0] - 1, try_pos[1]]
            if direction == 1:
                try_pos = [try_pos[0] - 1, try_pos[1] + 1]
            if direction == 2:
                try_pos = [try_pos[0], try_pos[1] + 1]
            if direction == 3:
                try_pos = [try_pos[0] + 1, try_pos[1] + 1]
            if direction == 4:
                try_pos = [try_pos[0] + 1, try_pos[1]]
            if direction == 5:
                try_pos = [try_pos[0] + 1, try_pos[1] - 1]
            if direction == 6:
                try_pos = [try_pos[0], try_pos[1] - 1]
            if direction == 7:
                try_pos = [try_pos[0] - 1, try_pos[1] - 1]
            hit = not self.check_target(try_pos) or count == ray_length

        return count

    def rotate(self, direction: int):
        self.orientation += direction
        if self.orientation < 0:
            self.orientation = self.max_orient
        if self.orientation > self.max_orient:
            self.orientation = 0

    def step(self, action: int):
        """
        Steps the environment forward given an action.
        """
        if self.orientation_type == OrientationType.variable:
            # 0 - Counter-clockwise rotation
            # 1 - Clockwise rotation
            # 2 - Forward movement
            if action == 0:
                self.rotate(-1)
            if action == 1:
                self.rotate(1)
            if action == 2:
                move_array = self.direction_map[self.orientation]
                self.move_agent(move_array)
            self.looking = self.orientation
        else:
            # 0 - Up
            # 1 - Right
            # 2 - Down
            # 3 - Left
            # 4 - Stay
            move_array = self.direction_map[action]
            self.looking = action
            self.move_agent(move_array)
        self.episode_time += 1
        if action == 4:
            reward = 0
        else:
            reward = self.time_penalty
        eval_pos = tuple(self.agent_pos)
        if eval_pos in self.reward_locs:
            reward += self.reward_locs[eval_pos]
            if self.terminate_on_reward:
                self.done = True
        return self.observation, reward, self.done, {}
