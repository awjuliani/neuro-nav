import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any


class Grid2DRenderer:
    # Color constants
    BACKGROUND_COLOR = (235, 235, 235)
    GRID_LINE_COLOR = (210, 210, 210)
    WALL_COLOR_INNER = (165, 165, 165)
    WALL_COLOR_OUTER = (125, 125, 125)
    POSITIVE_REWARD_FILL = (100, 100, 255)
    POSITIVE_REWARD_BORDER = (50, 50, 200)
    NEGATIVE_REWARD_FILL = (255, 100, 100)
    NEGATIVE_REWARD_BORDER = (200, 50, 50)
    KEY_FILL = (255, 215, 0)
    KEY_BORDER = (200, 160, 0)
    DOOR_FILL = (0, 150, 0)
    DOOR_BORDER = (0, 100, 0)
    WARP_FILL = (130, 0, 250)
    WARP_BORDER = (80, 0, 200)
    AGENT_COLOR = (0, 0, 0)
    TEMPLATE_COLOR = (150, 150, 150)

    def __init__(self, grid_size: int, block_size: int = 32, block_border: int = 3):
        self.grid_size = grid_size
        self.block_size = block_size
        self.block_border = block_border
        self.cached_image = None
        self.cached_objects = None
        self.cached_visible_walls = None
        self.img_size = self.block_size * self.grid_size

    def get_square_edges(
        self, y: int, x: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        true_start = self.block_border + 1
        block_end = self.block_size - self.block_border - 1

        x_unit = x * self.block_size
        y_unit = y * self.block_size

        return (
            (y_unit + true_start, x_unit + true_start),
            (y_unit + block_end, x_unit + block_end),
        )

    def make_base_image(
        self, blocks: List[Tuple[int, int]], visible_walls: bool
    ) -> np.ndarray:
        img = np.ones((self.img_size, self.img_size, 3), np.uint8)
        img[:] = self.BACKGROUND_COLOR

        self.render_gridlines(img)
        if visible_walls:
            self.render_walls(img, blocks)
        return img

    def render_gridlines(self, img: np.ndarray) -> np.ndarray:
        # Draw grid lines
        for i in range(0, self.img_size + 1, self.block_size):
            cv.line(img, (0, i), (self.img_size, i), self.GRID_LINE_COLOR, 1)
            cv.line(img, (i, 0), (i, self.img_size), self.GRID_LINE_COLOR, 1)

        # Draw final lines at rightmost and bottommost parts
        cv.line(
            img,
            (self.img_size - 1, 0),
            (self.img_size - 1, self.img_size - 1),
            self.GRID_LINE_COLOR,
            1,
        )
        cv.line(
            img,
            (0, self.img_size - 1),
            (self.img_size - 1, self.img_size - 1),
            self.GRID_LINE_COLOR,
            1,
        )
        return img

    def render_walls(
        self, img: np.ndarray, blocks: List[Tuple[int, int]]
    ) -> np.ndarray:
        for y, x in blocks:
            start, end = self.get_square_edges(x, y)
            cv.rectangle(img, start, end, self.WALL_COLOR_INNER, -1)
            cv.rectangle(img, start, end, self.WALL_COLOR_OUTER, self.block_border - 1)
        return img

    def render_rewards(
        self,
        img: np.ndarray,
        rewards: Dict[Tuple[int, int], Any],
        terminate_on_reward: bool,
    ) -> None:
        for pos, reward in rewards.items():
            draw, factor, reward_value = self._process_reward(
                reward, terminate_on_reward
            )
            if draw:
                self._draw_reward(img, pos, factor, reward_value)

    def _draw_reward(
        self, img: np.ndarray, pos: Tuple[int, int], factor: float, reward_value: float
    ) -> None:
        fill_color, border_color = self._get_reward_colors(reward_value)
        start, end = self.get_square_edges(pos[1], pos[0])
        size_reduction = int(2 * factor)
        adjusted_start = (start[0] + size_reduction, start[1] + size_reduction)
        adjusted_end = (end[0] - size_reduction, end[1] - size_reduction)
        cv.rectangle(img, adjusted_start, adjusted_end, fill_color, -1)
        cv.rectangle(
            img, adjusted_start, adjusted_end, border_color, self.block_border - 1
        )

    def _process_reward(
        self, reward: Any, terminate_on_reward: bool
    ) -> Tuple[bool, float, float]:
        if isinstance(reward, list):
            draw = reward[1]
            factor = 1 if reward[2] else 1.5
            reward_value = reward[0]
        else:
            draw = True
            factor = 1 if terminate_on_reward else 1.5
            reward_value = reward
        return draw, factor, reward_value

    def _get_reward_colors(
        self, reward: float
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        return (
            (self.POSITIVE_REWARD_FILL, self.POSITIVE_REWARD_BORDER)
            if reward > 0
            else (self.NEGATIVE_REWARD_FILL, self.NEGATIVE_REWARD_BORDER)
        )

    def render_markers(
        self,
        img: np.ndarray,
        markers: Dict[Tuple[int, int], Tuple[float, float, float]],
    ) -> None:
        for pos, marker_col in markers.items():
            fill_color = tuple(int(np.clip(c, 0, 1) * 255) for c in marker_col)
            start, end = self.get_square_edges(pos[1], pos[0])
            cv.rectangle(img, start, end, fill_color, -1)

    def render_keys(self, img: np.ndarray, keys: List[Tuple[int, int]]) -> None:
        for key in keys:
            center = (
                key[1] * self.block_size + self.block_size // 2,
                key[0] * self.block_size + self.block_size // 2,
            )
            pts = np.array(
                [
                    [center[0], center[1] - 4],
                    [center[0] + 4, center[1]],
                    [center[0], center[1] + 4],
                    [center[0] - 4, center[1]],
                ],
                np.int32,
            )
            cv.fillPoly(img, [pts], self.KEY_FILL)
            cv.polylines(img, [pts], True, self.KEY_BORDER, 1)

    def render_doors(self, img: np.ndarray, doors: Dict[Tuple[int, int], str]) -> None:
        for pos, dir in doors.items():
            start, end = self.get_square_edges(pos[1], pos[0])
            if dir == "h":
                start = (start[0] - 2, start[1] + 5)
                end = (end[0] + 2, end[1] - 5)
            elif dir == "v":
                start = (start[0] + 5, start[1] - 2)
                end = (end[0] - 5, end[1] + 2)
            else:
                raise ValueError("Invalid door direction")
            cv.rectangle(img, start, end, self.DOOR_FILL, -1)
            cv.rectangle(img, start, end, self.DOOR_BORDER, self.block_border - 1)

    def render_warps(self, img: np.ndarray, warps: Dict[Tuple[int, int], Any]) -> None:
        for pos in warps.keys():
            start, _ = self.get_square_edges(pos[1], pos[0])
            center = (start[0] + 7, start[1] + 7)
            cv.circle(img, center, 8, self.WARP_FILL, -1)
            cv.circle(img, center, 8, self.WARP_BORDER, self.block_border - 1)

    def render_agent(
        self, img: np.ndarray, agent_pos: Tuple[int, int], agent_dir: int
    ) -> None:
        agent_dir = agent_dir if isinstance(agent_dir, int) else agent_dir.item()
        agent_size = self.block_size // 2
        agent_offset = self.block_size // 4
        x_offset = agent_pos[1] * self.block_size + agent_offset
        y_offset = agent_pos[0] * self.block_size + agent_offset

        triangle_pts = {
            0: [(0, 1), (1, 1), (0.5, 0)],  # facing up
            1: [(0, 0), (0, 1), (1, 0.5)],  # facing right
            2: [(0, 0), (1, 0), (0.5, 1)],  # facing down
            3: [(1, 0), (1, 1), (0, 0.5)],  # facing left
        }

        pts = np.array(
            [
                (x_offset + pt[0] * agent_size, y_offset + pt[1] * agent_size)
                for pt in triangle_pts[agent_dir]
            ],
            dtype=np.int32,
        )

        cv.fillConvexPoly(img, pts, self.AGENT_COLOR)

    def render_frame(self, env: Any) -> np.ndarray:
        if self._should_update_cache(env):
            self._update_cache(env)
            img = self._create_new_frame(env)
            self.cached_image = img.copy()
        else:
            img = self.cached_image.copy()

        self.render_agent(img, env.agent_pos, env.looking)
        return img

    def _should_update_cache(self, env: Any) -> bool:
        objects_changed = self.cached_objects != env.objects
        blocks_changed = (
            self.cached_objects is None or env.blocks != self.cached_objects[1]
        )
        visible_walls_changed = self.cached_visible_walls != env.visible_walls
        return objects_changed or blocks_changed or visible_walls_changed

    def _update_cache(self, env: Any) -> None:
        self.cached_objects = [env.objects.copy(), env.blocks.copy()]
        self.cached_visible_walls = env.visible_walls

    def _create_new_frame(self, env: Any) -> np.ndarray:
        img = self.make_base_image(env.blocks, env.visible_walls)
        self.render_rewards(img, env.objects["rewards"], env.terminate_on_reward)
        self.render_markers(img, env.objects["markers"])
        self.render_keys(img, env.objects["keys"])
        self.render_doors(img, env.objects["doors"])
        self.render_warps(img, env.objects["warps"])
        return img

    def render_window(
        self, env: Any, w_size: int = 2, resize: bool = True
    ) -> np.ndarray:
        base_image = self.render_frame(env)
        template_size = self.block_size * (self.grid_size + 2)
        template = (
            np.ones((template_size, template_size, 3), dtype=np.uint8)
            * self.TEMPLATE_COLOR
        )
        template[
            self.block_size : -self.block_size, self.block_size : -self.block_size
        ] = base_image

        x, y = env.agent_pos
        window = template[
            self.block_size * (x - w_size + 1) : self.block_size * (x + w_size + 2),
            self.block_size * (y - w_size + 1) : self.block_size * (y + w_size + 2),
        ]

        return (
            cv.resize(window, (64, 64), interpolation=cv.INTER_NEAREST)
            if resize
            else window
        )

    def render(self, env: Any, mode: str = "human") -> np.ndarray:
        img = self.render_frame(env)
        if mode == "human":
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        return img
