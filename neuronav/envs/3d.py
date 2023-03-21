import numpy as np
from neuronav.envs.grid_env import GridEnv, GridOrientation, GridTemplate
import cv2
from cv2 import putText, FONT_HERSHEY_SIMPLEX


sky_color = (200, 100, 50)
wall_color = (100, 100, 100)

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
        orientation_type=GridOrientation.variable, template=GridTemplate.four_rooms
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


def render_walls(view, wall_list, player_pos, player_angle, wall_texture):
    wall_height = 100  # Height of the walls

    x_coords = np.arange(SCREEN_WIDTH)
    ray_angles = (2 * x_coords / float(SCREEN_WIDTH) - 1) * FOV

    ray_dir_x = np.sin(player_angle + ray_angles)
    ray_dir_y = np.cos(player_angle + ray_angles)

    # Raycast to find visible walls
    for i in range(NUM_RAYS):
        # Initialize step sizes for ray
        delta_x = np.abs(1 / ray_dir_x[i])
        delta_y = np.abs(1 / ray_dir_y[i])

        # Initialize wall_hit variable
        wall_hit = False

        # Calculate step direction and initial side distances
        if ray_dir_x[i] < 0:
            step_x = -1
            side_dist_x = (player_pos[1] - np.floor(player_pos[1])) * delta_x
        else:
            step_x = 1
            side_dist_x = (np.ceil(player_pos[1]) - player_pos[1]) * delta_x

        if ray_dir_y[i] < 0:
            step_y = -1
            side_dist_y = (player_pos[0] - np.floor(player_pos[0])) * delta_y
        else:
            step_y = 1
            side_dist_y = (np.ceil(player_pos[0]) - player_pos[0]) * delta_y

        # Perform DDA algorithm to find wall
        map_pos = np.array([int(player_pos[0]), int(player_pos[1])], dtype=int)
        side = None
        while not wall_hit:
            # Jump to the next square along the ray
            if side_dist_x < side_dist_y:
                side_dist_x += delta_x
                map_pos[1] += step_x
                side = 0
            else:
                side_dist_y += delta_y
                map_pos[0] += step_y
                side = 1

            # Check if the current square contains a wall
            if [map_pos[0], map_pos[1]] in wall_list:
                wall_hit = True

        # Calculate distance to the wall
        if side == 0:
            perp_wall_dist = np.abs(
                (map_pos[1] - player_pos[1] + step_x * 0.5) / ray_dir_x[i]
            )
        else:
            perp_wall_dist = np.abs(
                (map_pos[0] - player_pos[0] + step_y * 0.5) / ray_dir_y[i]
            )

        # Calculate the height of the wall strip on screen
        if perp_wall_dist == 0:
            perp_wall_dist = 0.5
        # Fisheye correction
        corrected_dist = perp_wall_dist * np.cos(ray_angles[i])
        wall_strip_height = int(SCREEN_HEIGHT / corrected_dist)

        # Calculate the position of the wall strip on screen
        draw_start = max(0, (SCREEN_HEIGHT - wall_strip_height) // 2)
        draw_end = min(SCREEN_HEIGHT, (SCREEN_HEIGHT + wall_strip_height) // 2)

        # Texture mapping
        if side == 0:
            wall_x = player_pos[0] + perp_wall_dist * ray_dir_y[i]
        else:
            wall_x = player_pos[1] + perp_wall_dist * ray_dir_x[i]
        wall_x -= int(wall_x)

        # Get the x-coordinate of the texture
        tex_x = int(wall_x * float(wall_texture.shape[1]))
        if side == 0 and ray_dir_x[i] > 0:
            tex_x = wall_texture.shape[1] - tex_x - 1
        if side == 1 and ray_dir_y[i] < 0:
            tex_x = wall_texture.shape[1] - tex_x - 1

        # Draw the wall strip with texture
        for j in range(draw_start, draw_end):
            tex_y = ((j - draw_start) * wall_texture.shape[0]) // (
                draw_end - draw_start
            )
            color = wall_texture[
                tex_y % wall_texture.shape[0], tex_x % wall_texture.shape[1]
            ]
            view[j, i] = color
    return view


def render_pseudo_3d_view(
    gridworld, player_pos, player_angle, floor_texture, wall_texture
):
    view = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)

    # Draw sky
    view = render_sky(view)

    # Draw floor
    view = render_floor(view, player_pos, player_angle, floor_texture)

    # Draw walls
    wall_list = gridworld.blocks
    view = render_walls(view, wall_list, player_pos, player_angle, wall_texture)

    view = cv2.flip(view, 1)
    return view


def main():
    floor_texture = cv2.imread("./neuronav/envs/floor_texture.jpg")
    # downsample the texture to speed up rendering
    floor_texture = cv2.resize(
        floor_texture, (floor_texture.shape[1] // 8, floor_texture.shape[0] // 8)
    )
    wall_texture = cv2.imread("./neuronav/envs/wall_texture.jpg")
    # downsample the texture to speed up rendering
    wall_texture = cv2.resize(
        wall_texture, (wall_texture.shape[1] // 8, wall_texture.shape[0] // 8)
    )

    gridworld = create_gridworld()

    while True:
        player_pos, player_angle = (
            gridworld.agent_pos,
            (gridworld.looking / 4) * -np.pi * 2 + np.pi,
        )
        top_down_view = render_top_down_view(gridworld)
        pseudo_3d_view = render_pseudo_3d_view(
            gridworld, player_pos, player_angle, floor_texture, wall_texture
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
