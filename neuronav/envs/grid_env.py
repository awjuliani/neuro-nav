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
    dynamic = "dynamic"


class GridEnv(Env):
    """
    Grid Environment
    """

    def __init__(
        self,
        topography=GridTopography.empty,
        grid_size=GridSize.small,
        obs_type=GridObsType.index,
        orientation_type=OrientationType.fixed,
    ):
        self.blocks, self.agent_start_pos, self.topo_reward_locs = generate_topography(
            topography, grid_size
        )
        self.grid_size = grid_size.value
        self.state_size = self.grid_size * self.grid_size
        self.orientation_type = orientation_type
        self.max_orient = 3
        if self.orientation_type == OrientationType.dynamic:
            self.action_space = spaces.Discrete(3)
            self.orient_size = 4
        elif self.orientation_type == OrientationType.fixed:
            self.orient_size = 1
            self.action_space = spaces.Discrete(4)
        else:
            raise Exception("No valid OrientationType provided.")
        self.state_size *= self.orient_size
        self.agent_pos = []
        self.direction_map = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
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
                    self.grid_size,
                    self.grid_size,
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
                    5,
                    5,
                    3,
                ),
            )
        else:
            raise Exception("No valid ObservationType provided.")

    def reset(
        self,
        reward_locs=None,
        agent_pos=None,
        episode_length=100,
        random_start=False,
    ):
        self.done = False
        self.episode_time = 0
        self.orientation = 0
        self.max_episode_time = episode_length

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

    def grid(self, render_objects=True):
        grid = np.zeros([self.grid_size, self.grid_size, 3])
        if render_objects:
            grid[self.agent_pos[0], self.agent_pos[1], :] = 1
            for loc, reward in self.reward_locs.items():
                if reward > 0:
                    grid[loc[0], loc[1], 1] = reward
                else:
                    grid[loc[0], loc[1], 0] = np.abs(reward)
        for block in self.blocks:
            grid[block[0], block[1], :] = 0.5
        return grid

    def render(self):
        grid = self.grid()
        if self.orientation_type == OrientationType.dynamic:
            grid[self.agent_pos[0], self.agent_pos[1], :] = 0
        plt.imshow(grid)
        if self.orientation_type == OrientationType.dynamic:
            up = [[0.5, 0], [0, 1], [1, 1]]
            right = [[0, 0], [1, 0.5], [0, 1]]
            down = [[0, 0], [0.5, 1], [1, 0]]
            left = [[0, 0.5], [1, 1], [1, 0]]
            arrow_list = [up, right, down, left]
            selection = arrow_list[self.orientation]
            for i in range(3):
                selection[i][1] += self.agent_pos[0] - 0.5
                selection[i][0] += self.agent_pos[1] - 0.5
            t1 = plt.Polygon(selection, color="white")
            plt.gca().add_patch(t1)
        plt.axis("off")
        plt.hlines(
            y=np.arange(0, self.grid_size) + 0.5,
            xmin=np.full(self.grid_size, 0) - 0.5,
            xmax=np.full(self.grid_size, self.grid_size) - 0.5,
            color="dimgray",
            linewidths=1.0,
        )
        plt.vlines(
            x=np.arange(0, self.grid_size) + 0.5,
            ymin=np.full(self.grid_size, 0) - 0.5,
            ymax=np.full(self.grid_size, self.grid_size) - 0.5,
            color="dimgray",
            linewidth=1.0,
        )
        plt.show()

    def move_agent(self, direction):
        new_pos = self.agent_pos + direction
        if self.check_target(new_pos):
            self.agent_pos = list(new_pos)

    def check_target(self, target):
        x_check = -1 < target[0] < self.grid_size
        y_check = -1 < target[1] < self.grid_size
        block_check = list(target) not in self.blocks
        return x_check and y_check and block_check

    def get_observation(self, perspective):
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
            if self.orientation_type == OrientationType.dynamic:
                two_hot = np.concatenate(
                    [two_hot, utils.onehot(self.orientation, self.orient_size)]
                )
            return two_hot
        elif self.obs_mode == GridObsType.geometric:
            geo = np.array(perspective) / (self.grid_size - 1.0)
            if self.orientation_type == OrientationType.dynamic:
                geo = np.concatenate(
                    [geo, utils.onehot(self.orientation, self.orient_size)]
                )
            return geo
        elif self.obs_mode == GridObsType.visual:
            return self.grid()
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
            if self.orientation_type == OrientationType.dynamic:
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
            space = np.ones([self.grid_size * 3, self.grid_size * 3, 3]) * 0.5
            space[
                self.grid_size : self.grid_size * 2,
                self.grid_size : self.grid_size * 2,
                :,
            ] = self.grid()
            x, y = self.agent_pos
            x += self.grid_size
            y += self.grid_size
            sub = space[x - 2 : x + 3, y - 2 : y + 3, :]
            return sub

    @property
    def observation(self):
        return self.get_observation(self.agent_pos)

    def get_boundaries(
        self,
        object_point,
        use_onehot=False,
        num_rays=4,
        ray_length=10,
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

    def simple_ray(self, direction, start):
        if self.orientation_type == OrientationType.dynamic:
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

    def rotate(self, direction):
        self.orientation += direction
        if self.orientation < 0:
            self.orientation = self.max_orient
        if self.orientation > self.max_orient:
            self.orientation = 0

    def step(self, action):
        if self.orientation_type == OrientationType.dynamic:
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
        else:
            # 0 - Up
            # 1 - Down
            # 2 - Left
            # 3 - Right
            move_array = self.direction_map[action]
            self.move_agent(move_array)
        self.episode_time += 1
        reward = 0.0
        eval_pos = tuple(self.agent_pos)
        if eval_pos in self.reward_locs:
            if np.abs(self.reward_locs[eval_pos]) == 1.0:
                self.done = True
            reward = self.reward_locs[eval_pos]
        return self.observation, reward, self.done, {}
