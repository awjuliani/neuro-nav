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
import matplotlib.pyplot as plt
import cv2 as cv
import copy
from neuronav.envs.grid_3d import Grid3DRenderer


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


class GridOrientation(enum.Enum):
    fixed = "fixed"
    variable = "variable"


class GridEnv(Env):
    """
    Grid Environment. A 2D maze-like OpenAI gym compatible RL environment.
    """

    def __init__(
        self,
        template: GridTemplate = GridTemplate.empty,
        size: GridSize = GridSize.small,
        obs_type: GridObservation = GridObservation.index,
        orientation_type: GridOrientation = GridOrientation.fixed,
        seed: int = None,
        use_noop: bool = False,
    ):
        self.rng = np.random.RandomState(seed)
        self.use_noop = use_noop
        self.blocks, self.agent_start_pos, self.template_objects = generate_layout(
            template, size
        )
        self.grid_size = size.value
        self.state_size = self.grid_size * self.grid_size
        self.orientation_type = orientation_type
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
            self.action_space = spaces.Discrete(3 + self.use_noop)
            self.orient_size = 4
        elif self.orientation_type == GridOrientation.fixed:
            self.orient_size = 1
            self.action_space = spaces.Discrete(4 + self.use_noop)
        else:
            raise Exception("No valid GridOrientation provided.")
        self.state_size *= self.orient_size

    def set_obs_space(self, obs_type):
        if isinstance(obs_type, str):
            obs_type = GridObservation(obs_type)
        self.obs_mode = obs_type
        if obs_type == GridObservation.visual:
            self.obs_space = spaces.Box(0, 1, shape=(128, 128, 3))
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
            self.obs_space = spaces.Box(0, 1, shape=(64, 64, 3))
        elif obs_type == GridObservation.window_tight:
            self.obs_space = spaces.Box(0, 1, shape=(64, 64, 3))
        elif obs_type == GridObservation.symbolic:
            self.obs_space = spaces.Box(0, 1, shape=(self.grid_size, self.grid_size, 6))
        elif obs_type == GridObservation.symbolic_window:
            self.obs_space = spaces.Box(0, 1, shape=(5, 5, 6))
        elif obs_type == GridObservation.symbolic_window_tight:
            self.obs_space = spaces.Box(0, 1, shape=(3, 3, 6))
        elif obs_type == GridObservation.rendered_3d:
            self.obs_space = spaces.Box(0, 1, shape=(128, 128, 3))
            self.renderer = Grid3DRenderer(128)
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

    def symbolic_obs(self):
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
        grid[tuple(self.agent_pos), 0] = 1

        # Set rewards
        reward_locs = [
            loc
            for loc, reward in self.objects["rewards"].items()
            if type(reward) != list or reward[1] == 1
        ]
        grid[tuple(zip(*reward_locs)), 1] = 1

        # Set keys
        key_locs = self.objects["keys"]
        grid[tuple(zip(*key_locs)), 2] = 1

        # Set doors
        door_locs = self.objects["doors"]
        grid[tuple(zip(*door_locs)), 3] = 1

        # Set warps
        warp_locs = self.objects["warps"].keys()
        grid[tuple(zip(*warp_locs)), 5] = 1

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

    def symbolic_window_obs(self, size: int = 5):
        if size not in [3, 5]:
            raise ValueError("Window size must be 3 or 5")

        obs = self.symbolic_obs()
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

    def render(self, provide=False):
        """
        Renders the environment in a pyplot window.
        """
        image = self.make_visual_obs()
        if self.obs_mode == GridObservation.rendered_3d:
            img_first = self.renderer.render_frame(self)
            top_down = cv.resize(image, (128, 128))
            image = np.concatenate((img_first, top_down), axis=1)
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        if provide:
            return image

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
        if self.visible_walls:
            # draw the blocks
            for x, y in self.blocks:
                start, end = self.get_square_edges(x, y, block_size, block_size - 2)
                cv.rectangle(img, start, end, (175, 175, 175), -1)
                cv.rectangle(img, start, end, (125, 125, 125), block_border - 1)
        # draw the reward locations
        for pos, reward in self.objects["rewards"].items():
            if type(reward) != list:
                draw = True
                if self.terminate_on_reward:
                    factor = 1
                else:
                    factor = 1.5
            elif reward[1] == True:
                draw = True
                if reward[2]:
                    factor = 1
                else:
                    factor = 1.5
                reward = reward[0]
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
                    pos[0], pos[1], block_size, block_size - int(4 * factor)
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
            # generate a diamond shape for the key
            pts = np.array(
                [
                    [
                        key[1] * block_size + block_size // 2,
                        key[0] * block_size + block_size // 2 - 4,
                    ],
                    [
                        key[1] * block_size + block_size // 2 + 4,
                        key[0] * block_size + block_size // 2,
                    ],
                    [
                        key[1] * block_size + block_size // 2,
                        key[0] * block_size + block_size // 2 + 4,
                    ],
                    [
                        key[1] * block_size + block_size // 2 - 4,
                        key[0] * block_size + block_size // 2,
                    ],
                ]
            )

            cv.fillPoly(img, [pts], fill_color)
            cv.polylines(img, [pts], True, border_color, 1)

        # draw the doors
        for pos, dir in self.objects["doors"].items():
            fill_color = (0, 150, 0)
            border_color = (0, 100, 0)
            start, end = self.get_square_edges(
                pos[0], pos[1], block_size, block_size - 2
            )
            if dir == "h":
                start = (start[0] - 2, start[1] + 5)
                end = (end[0] + 2, end[1] - 5)
            elif dir == "v":
                start = (start[0] + 5, start[1] - 2)
                end = (end[0] - 5, end[1] + 2)
            else:
                raise ValueError("Invalid door direction")
            cv.rectangle(img, start, end, fill_color, -1)
            cv.rectangle(img, start, end, border_color, block_border - 1)

        # draw the warp locations. They are purple
        for pos, target in self.objects["warps"].items():
            fill_color = (130, 0, 250)
            border_color = (80, 0, 200)
            start, end = self.get_square_edges(
                pos[0], pos[1], block_size, block_size - 2
            )
            # draw a circle at the warp pos
            cv.circle(img, (start[0] + 7, start[1] + 7), 8, fill_color, -1)
            cv.circle(
                img, (start[0] + 7, start[1] + 7), 8, border_color, block_border - 1
            )

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
            img = cv.resize(img, (128, 128))
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
        door_check = tuple(target) not in self.objects["doors"].keys()
        if self.keys > 0 and door_check is False:
            door_check = True
            self.objects["doors"].pop(tuple(target))
            self.keys -= 1
        return x_check and y_check and block_check and door_check

    def get_observation(self, perspective: list):
        """
        Returns an observation corresponding to the provided coordinates.
        """
        if self.obs_mode == GridObservation.onehot:
            # one-hot encoding of the perspective
            one_hot = utils.onehot(
                self.orientation * self.grid_size * self.grid_size
                + perspective[0] * self.grid_size
                + perspective[1],
                self.state_size * self.orient_size,
            )
            return one_hot
        elif self.obs_mode == GridObservation.twohot:
            two_hot = utils.twohot(perspective, self.grid_size)
            if self.orientation_type == GridOrientation.variable:
                two_hot = np.concatenate(
                    [two_hot, utils.onehot(self.orientation, self.orient_size)]
                )
            return two_hot
        elif self.obs_mode == GridObservation.geometric:
            geo = np.array(perspective) / (self.grid_size - 1.0)
            if self.orientation_type == GridOrientation.variable:
                geo = np.concatenate(
                    [geo, utils.onehot(self.orientation, self.orient_size)]
                )
            return geo
        elif self.obs_mode == GridObservation.visual:
            return self.make_visual_obs(True)
        elif self.obs_mode == GridObservation.index:
            idx = (
                self.orientation * self.grid_size * self.grid_size
                + perspective[0] * self.grid_size
                + perspective[1]
            )
            return idx
        elif self.obs_mode == GridObservation.boundary:
            bounds = self.get_boundaries(
                perspective, False, self.num_rays, self.ray_length
            )
            if self.orientation_type == GridOrientation.variable:
                bounds = np.concatenate(
                    [bounds, utils.onehot(self.orientation, self.orient_size)]
                )
            return bounds
        elif self.obs_mode == GridObservation.images:
            idx = (
                self.orientation * self.state_size
                + perspective[0] * self.grid_size
                + perspective[1]
            )
            return np.rot90(self.images[idx], k=3)
        elif self.obs_mode == GridObservation.window:
            return self.make_window()
        elif self.obs_mode == GridObservation.symbolic:
            return self.symbolic_obs()
        elif self.obs_mode == GridObservation.symbolic_window:
            return self.symbolic_window_obs()
        elif self.obs_mode == GridObservation.window_tight:
            return self.make_window(w_size=1)
        elif self.obs_mode == GridObservation.symbolic_window_tight:
            return self.symbolic_window_obs(size=3)
        elif self.obs_mode == GridObservation.rendered_3d:
            return self.renderer.render_frame(self)
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

        if self.orientation_type == GridOrientation.variable:
            # 0 - Counter-clockwise rotation
            # 1 - Clockwise rotation
            # 2 - Forward movement
            if action == 0:
                self.rotate(-1)
            elif action == 1:
                self.rotate(1)
            elif action == 2:
                move_array = self.direction_map[self.orientation]
                self.move_agent(move_array)
            self.looking = self.orientation
        else:
            # 0 - North
            # 1 - East
            # 2 - South
            # 3 - West
            # 4 - Stay
            move_array = self.direction_map[action]
            self.looking = action
            self.move_agent(move_array)

        self.episode_time += 1
        reward = 0 if action == 4 else self.time_penalty
        eval_pos = tuple(self.agent_pos)
        terminate = self.terminate_on_reward

        if eval_pos in self.objects["rewards"]:
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
            self.renderer.close()
        return super().close()
