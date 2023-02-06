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
import copy


class GridObsType(enum.Enum):
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
        seed: int = None,
    ):
        self.rng = np.random.RandomState(seed)
        self.blocks, self.agent_start_pos, self.topo_objects = generate_topography(
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
        self.base_objects = {
            "rewards": {},
            "markers": {},
            "keys": [],
            "doors": [],
            "warps": {},
        }
        self.direction_map = np.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0]])
        self.done = False
        self.keys = 0
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
        elif obs_type == GridObsType.symbolic:
            self.observation_space = spaces.Box(
                0,
                1,
                shape=(
                    self.grid_size,
                    self.grid_size,
                    6,
                ),
            )
        elif obs_type == GridObsType.symbolic_window:
            self.observation_space = spaces.Box(
                0,
                1,
                shape=(
                    5,
                    5,
                    6,
                ),
            )
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

        if agent_pos != None:
            self.agent_pos = agent_pos
        elif random_start:
            self.agent_pos = self.get_free_spot()
        else:
            self.agent_pos = self.agent_start_pos

        base_object = copy.deepcopy(self.base_objects)
        if objects != None:
            use_objects = copy.deepcopy(objects)
        else:
            use_objects = copy.deepcopy(self.topo_objects)
        for key in use_objects.keys():
            if key in base_object.keys():
                base_object[key] = use_objects[key]
        self.objects = base_object
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

    def symbolic_obs(self):
        """
        Returns a symbolic representation of the environment in a numpy tensor.
        Tensor shape is (grid_size, grid_size, 5)
        5 channels are:
            0: agent
            1: rewards
            2: keys
            3: doors
            4: walls
        """
        grid = np.zeros([self.grid_size, self.grid_size, 6])
        grid[self.agent_pos[0], self.agent_pos[1], 0] = 1
        for loc, reward in self.objects["rewards"].items():
            if type(reward) != list:
                draw = True
            elif reward[1] == 1:
                draw = True
            else:
                draw = False
            if draw:
                grid[loc[0], loc[1], 1] = 1
        for loc in self.objects["keys"]:
            grid[loc[0], loc[1], 2] = 1
        for loc in self.objects["doors"]:
            grid[loc[0], loc[1], 3] = 1
        for loc in self.objects["warps"].keys():
            grid[loc[0], loc[1], 5] = 1
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

    def symbolic_window_obs(self):
        # return a 5x5x5 tensor of the surrounding area
        # Pads the edges with walls if the agent is near the edge
        obs = self.symbolic_obs()
        full_window = np.zeros([self.grid_size + 2, self.grid_size + 2, 6])
        full_window[1:-1, 1:-1, :] = obs
        full_window[0, :, 4] = 1
        full_window[:, 0, 4] = 1
        full_window[-1, :, 4] = 1
        full_window[:, -1, 4] = 1
        window = full_window[
            self.agent_pos[0] - 1 : self.agent_pos[0] + 4,
            self.agent_pos[1] - 1 : self.agent_pos[1] + 4,
            :,
        ]
        return window

    def render(self):
        """
        Renders the environment in a pyplot window.
        """
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
        """
        Returns a visual observation of the environment from a top-down perspective.
        """
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
            cv.rectangle(img, start, end, (175, 175, 175), -1)
            cv.rectangle(img, start, end, (125, 125, 125), block_border - 1)
        # draw the reward locations
        for pos, reward in self.objects["rewards"].items():
            if type(reward) != list:
                draw = True
            elif reward[1] == 1:
                draw = True
            else:
                draw = False
            if draw:
                if reward > 0:
                    fill_color = (100, 100, 255)  # blue
                    border_color = (50, 50, 200)  # blue
                else:
                    fill_color = (255, 100, 100)  # red
                    border_color = (200, 50, 50)  # red
                start, end = self.get_square_edges(
                    pos[0], pos[1], block_size, block_size - 4
                )
                cv.rectangle(img, start, end, fill_color, -1)
                cv.rectangle(img, start, end, border_color, block_border - 1)

        # draw the markers
        for pos, marker_col in self.objects["markers"].items():
            fill_color = marker_col
            # clamp fill colors between and 0 and 1 and multiply by 255
            fill_color = list(fill_color)
            for i in range(3):
                fill_color[i] = np.clip(fill_color[i], 0, 1).item() * 255
            fill_color = tuple(fill_color)
            start, end = self.get_square_edges(
                pos[0], pos[1], block_size, block_size - 1
            )
            cv.rectangle(img, start, end, fill_color, -1)

        # draw the keys
        for key in self.objects["keys"]:
            fill_color = (255, 215, 0)
            border_color = (200, 160, 0)
            start, end = self.get_square_edges(
                key[0], key[1], block_size, block_size - 5
            )
            cv.rectangle(img, start, end, fill_color, -1)
            cv.rectangle(img, start, end, border_color, block_border - 1)

        # draw the doors
        for pos in self.objects["doors"]:
            fill_color = (0, 150, 0)
            border_color = (0, 100, 0)
            start, end = self.get_square_edges(
                pos[0], pos[1], block_size, block_size - 2
            )
            cv.rectangle(img, start, end, fill_color, -1)
            cv.rectangle(img, start, end, border_color, block_border - 1)

        # draw the warp locations. They are purple
        for pos, target in self.objects["warps"].items():
            fill_color = (130, 0, 250)
            border_color = (80, 0, 200)
            start, end = self.get_square_edges(
                pos[0], pos[1], block_size, block_size - 2
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
        """
        Returns a window of size (w_size * 2 + 1) x (w_size * 2 + 1) around the agent.
        The window is padded with 1 block on each side to account for the agent's
        position.
        """
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
        x_check = -1 < target[0] < self.grid_size
        y_check = -1 < target[1] < self.grid_size
        block_check = list(target) not in self.blocks
        door_check = tuple(target) not in self.objects["doors"]
        if self.keys > 0 and door_check is False:
            door_check = True
            self.objects["doors"].remove(tuple(target))
            self.keys -= 1
        return x_check and y_check and block_check and door_check

    def get_observation(self, perspective: list):
        """
        Returns an observation corresponding to the provided coordinates.
        """
        if self.obs_mode == GridObsType.onehot:
            # one-hot encoding of the perspective
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
        elif self.obs_mode == GridObsType.symbolic:
            return self.symbolic_obs()
        elif self.obs_mode == GridObsType.symbolic_window:
            return self.symbolic_window_obs()
        else:
            raise ValueError("Invalid observation mode.")

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
        """
        Returns the distance to the nearest object in the given direction.
        """
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
        """
        Rotates the agent orientation in the given direction.
        """
        self.orientation += direction
        if self.orientation < 0:
            self.orientation = self.max_orient
        if self.orientation > self.max_orient:
            self.orientation = 0

    def step(self, action: int):
        """
        Steps the environment forward given an action.
        Action is an integer in the range [0, self.action_space.n).
        """
        if self.stochasticity > self.rng.rand():
            action = self.rng.randint(0, self.action_space.n)
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
        if eval_pos in self.objects["rewards"]:
            loc_reward = self.objects["rewards"][eval_pos]
            if type(loc_reward) == list:
                loc_reward = loc_reward[0]
            reward += loc_reward
            if self.terminate_on_reward:
                self.done = True
            self.objects["rewards"].pop(eval_pos)
        if eval_pos in self.objects["keys"]:
            self.keys += 1
            self.objects["keys"].remove(eval_pos)
        if eval_pos in self.objects["warps"]:
            self.agent_pos = self.objects["warps"][eval_pos]
        return self.observation, reward, self.done, {}
