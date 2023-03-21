import numpy as np
from neuronav.envs.grid_env import GridEnv, GridOrientation, GridTemplate
import cv2

sky_color = (200, 100, 50)

# Pseudo-3D rendering parameters
SCREEN_WIDTH, SCREEN_HEIGHT = 480, 480
FOV = np.pi / 3
NUM_RAYS = 480

# Actions
MOVE_FORWARD = 2
TURN_RIGHT = 1
TURN_LEFT = 0


def create_gridworld():
    env = GridEnv(
        orientation_type=GridOrientation.variable, template=GridTemplate.obstacle
    )
    env.reset()
    return env


def render_top_down_view(gridworld):
    grid_img = gridworld.make_visual_obs()
    return cv2.resize(
        grid_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST
    )


def render_sky(view):
    view[: SCREEN_HEIGHT // 2, :, :] = sky_color
    return view


def render_floor(view, player_pos, player_angle, floor_texture):
    y_coords = np.arange(SCREEN_HEIGHT // 2 + 1, SCREEN_HEIGHT)
    p = SCREEN_HEIGHT / (2.0 * y_coords - SCREEN_HEIGHT)[:, np.newaxis]

    x_coords = np.arange(SCREEN_WIDTH)
    ray_angles = (2 * x_coords / float(SCREEN_WIDTH) - 1) * FOV

    ray_dir_x = np.sin(player_angle + ray_angles)
    ray_dir_y = np.cos(player_angle + ray_angles)

    floor_x = player_pos[1] + p * ray_dir_x
    floor_y = player_pos[0] + p * ray_dir_y

    tex_x = (floor_x * floor_texture.shape[1]).astype(int) % floor_texture.shape[1]
    tex_y = (floor_y * floor_texture.shape[0]).astype(int) % floor_texture.shape[0]
    view[SCREEN_HEIGHT // 2 + 1 :, :] = floor_texture[tex_y, tex_x]
    return view


def cast_ray(gridworld, player_pos, ray_angle):
    ray_pos, ray_dir = np.array(player_pos, dtype=float), np.array(
        [np.cos(ray_angle), np.sin(ray_angle)]
    )

    delta_dist = np.abs(1 / (ray_dir + 1e-8))
    side_dist = (ray_pos - np.floor(ray_pos)) * delta_dist

    step = np.sign(ray_dir)
    map_pos = np.floor(ray_pos).astype(int)

    hit = False
    while not hit:
        side = 0 if side_dist[0] < side_dist[1] else 1
        side_dist[side] += delta_dist[side]
        map_pos[side] += step[side]
        if [map_pos[0], map_pos[1]] in gridworld.blocks:
            hit = True

    # Calculate the distance to the wall, considering the correct wall location.
    distance = (map_pos[side] - ray_pos[side] + (1 - step[side]) / 2) / ray_dir[side]
    return distance


def get_wall_height(distance):
    max_wall_height = SCREEN_HEIGHT
    height_candidate = SCREEN_HEIGHT / distance
    if np.isinf(height_candidate):
        height_candidate = max_wall_height
    height = int(height_candidate)
    return height


def render_walls(view, gridworld, player_pos, player_angle):
    def get_wall_color(distance):
        max_distance = 10
        normalized_distance = np.clip(distance / max_distance, 0, 1)
        color_intensity = int(255 * (1 - normalized_distance))
        return (color_intensity, color_intensity, color_intensity)

    for i in range(NUM_RAYS):
        ray_angle = player_angle - (i / NUM_RAYS - 0.5) * FOV
        distance = cast_ray(gridworld, player_pos, ray_angle)
        height = get_wall_height(distance)

        x = i * (SCREEN_WIDTH) // NUM_RAYS
        wall_color = get_wall_color(distance)

        wall_start = (SCREEN_HEIGHT - height) // 2
        wall_end = (SCREEN_HEIGHT + height) // 2

        view[wall_start:wall_end, x] = wall_color

    return view


def render_pseudo_3d_view(
    gridworld, player_pos, player_angle, wall_texture, floor_texture
):
    view = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    # Draw sky
    view = render_sky(view)

    # Draw floor
    view = render_floor(view, player_pos, player_angle, floor_texture)

    # Draw walls
    view = render_walls(view, gridworld, player_pos, player_angle)

    return view


def main():
    wall_texture = cv2.imread("./notebooks/wall_texture.jpg")
    # downsample the texture to speed up rendering
    wall_texture = cv2.resize(
        wall_texture, (wall_texture.shape[1] // 8, wall_texture.shape[0] // 8)
    )

    floor_texture = cv2.imread("./notebooks/floor_texture.jpg")
    # downsample the texture to speed up rendering
    floor_texture = cv2.resize(
        floor_texture, (floor_texture.shape[1] // 8, floor_texture.shape[0] // 8)
    )
    gridworld = create_gridworld()

    while True:
        player_pos, player_angle = (
            gridworld.agent_pos,
            (gridworld.looking / 4) * -np.pi * 2 + np.pi,
        )
        top_down_view = render_top_down_view(gridworld)
        pseudo_3d_view = render_pseudo_3d_view(
            gridworld, player_pos, player_angle, wall_texture, floor_texture
        )

        frame = np.hstack((top_down_view, pseudo_3d_view))
        cv2.imshow("Pseudo-3D Renderer", frame)
        print("Player pos: {}, Player angle: {}".format(player_pos, player_angle))
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        elif key == ord("w"):
            gridworld.step(2)
        elif key == ord("d"):
            gridworld.step(1)
        elif key == ord("a"):
            gridworld.step(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
