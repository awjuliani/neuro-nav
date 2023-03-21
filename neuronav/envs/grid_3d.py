import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import time
import cv2
import math

from neuronav.envs.grid_env import GridEnv, GridOrientation, GridTemplate


def load_texture(filename):
    img = Image.open(filename)
    img = img.convert("RGB")

    img_data = np.array(list(img.getdata()), np.uint8)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        img.width,
        img.height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        img_data,
    )

    return texture_id


def render_plane(x, y, z, area, texture_id, repeat=4):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glPushMatrix()
    glTranslatef(x, y, z)
    glBegin(GL_QUADS)

    half_area = area / 2.0

    # Plane
    glTexCoord2f(0, 0)
    glVertex3f(-half_area, 0.0, half_area)
    glTexCoord2f(repeat, 0)
    glVertex3f(half_area, 0.0, half_area)
    glTexCoord2f(repeat, repeat)
    glVertex3f(half_area, 0.0, -half_area)
    glTexCoord2f(0, repeat)
    glVertex3f(-half_area, 0.0, -half_area)

    glEnd()
    glPopMatrix()

    glDisable(GL_TEXTURE_2D)


def render_cube(x, y, z, texture):
    glPushMatrix()
    glTranslatef(x, y, z)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture)
    glBegin(GL_QUADS)

    # Front face
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-0.5, -0.5, 0.5)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(0.5, -0.5, 0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(0.5, 0.5, 0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-0.5, 0.5, 0.5)

    # Back face
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-0.5, 0.5, -0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(0.5, 0.5, -0.5)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(0.5, -0.5, -0.5)

    # Left face
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-0.5, -0.5, 0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-0.5, 0.5, 0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-0.5, 0.5, -0.5)

    # Right face
    glTexCoord2f(1.0, 0.0)
    glVertex3f(0.5, -0.5, -0.5)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(0.5, -0.5, 0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(0.5, 0.5, 0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(0.5, 0.5, -0.5)

    # Top face
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-0.5, 0.5, -0.5)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-0.5, 0.5, 0.5)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(0.5, 0.5, 0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(0.5, 0.5, -0.5)

    # Bottom face
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-0.5, -0.5, 0.5)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(0.5, -0.5, 0.5)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(0.5, -0.5, -0.5)

    glEnd()
    glDisable(GL_TEXTURE_2D)
    glPopMatrix()


def render_sphere(x, y, z, radius, slices=16, stacks=16, texture=None):
    glPushMatrix()
    glTranslatef(x, y, z)
    if texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)

    # Draw the sphere using quad strips
    for i in range(stacks + 1):
        lat0 = math.pi * (-0.5 + (i - 1) / stacks)
        z0 = radius * math.sin(lat0)
        zr0 = radius * math.cos(lat0)

        lat1 = math.pi * (-0.5 + i / stacks)
        z1 = radius * math.sin(lat1)
        zr1 = radius * math.cos(lat1)

        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * (j - 1) / slices
            x = math.cos(lng)
            y = math.sin(lng)

            glNormal3f(x * zr0, y * zr0, z0)
            if texture is not None:
                glTexCoord2f(j / slices, (i - 1) / stacks)
            glVertex3f(x * zr0, y * zr0, z0)

            glNormal3f(x * zr1, y * zr1, z1)
            if texture is not None:
                glTexCoord2f(j / slices, i / stacks)
            glVertex3f(x * zr1, y * zr1, z1)
        glEnd()

    if texture is not None:
        glDisable(GL_TEXTURE_2D)

    glPopMatrix()


def set_camera(agent_pos, agent_dir):
    pos = (agent_pos[0], 0, agent_pos[1])
    offsets = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    offset = offsets[agent_dir]

    target = (agent_pos[0] + offset[0], 0, agent_pos[1] + offset[1])
    up = (0, 1, 0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(
        pos[0], pos[1], pos[2], target[0], target[1], target[2], up[0], up[1], up[2]
    )


def render_walls(blocks, agent_pos, agent_dir):
    for block in blocks:
        # check if the block is in the agent's field of view
        # use the agent's orientation to determine the field of view
        if agent_dir == 0:
            if block[0] > agent_pos[0]:
                continue
        elif agent_dir == 1:
            if block[1] < agent_pos[1]:
                continue
        elif agent_dir == 2:
            if block[0] < agent_pos[0]:
                continue
        elif agent_dir == 3:
            if block[1] > agent_pos[1]:
                continue
        render_cube(block[0], 0.0, block[1], wall_id)


def render_sprite(x, y, z, width, height, texture_id):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glPushMatrix()
    glTranslatef(x, y, z)

    # Align the sprite to face the camera
    modelview_matrix = glGetFloatv(GL_MODELVIEW_MATRIX)
    right = np.array([modelview_matrix[0], modelview_matrix[4], modelview_matrix[8]])
    up = np.array([modelview_matrix[1], modelview_matrix[5], modelview_matrix[9]])

    half_width = width / 2.0
    half_height = height / 2.0

    vertices = [
        (-half_width * right) - (half_height * up),
        (half_width * right) - (half_height * up),
        (half_width * right) + (half_height * up),
        (-half_width * right) + (half_height * up),
    ]

    glBegin(GL_QUADS)

    # Sprite quad
    glTexCoord2f(0, 0)
    glVertex3f(vertices[0][0], vertices[0][1], vertices[0][2])
    glTexCoord2f(1, 0)
    glVertex3f(vertices[1][0], vertices[1][1], vertices[1][2])
    glTexCoord2f(1, 1)
    glVertex3f(vertices[2][0], vertices[2][1], vertices[2][2])
    glTexCoord2f(0, 1)
    glVertex3f(vertices[3][0], vertices[3][1], vertices[3][2])

    glEnd()
    glPopMatrix()

    glDisable(GL_TEXTURE_2D)


def initialize_glfw(resolution):
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(resolution, resolution, "Offscreen", None, None)
    glfw.make_context_current(window)
    return window


def configure_opengl():
    glClearColor(135 / 255, 206 / 255, 235 / 255, 1.0)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, 1, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


def save_image(image_np, filename):
    image = Image.fromarray(image_np)
    image.save(filename)


RESOLUTION = 256

window = initialize_glfw(RESOLUTION)
configure_opengl()

width, height = glfw.get_framebuffer_size(window)
tex_folder = "./textures/"
floor_id = load_texture(f"{tex_folder}floor.png")
wall_id = load_texture(f"{tex_folder}wall.png")
gem_id = load_texture(f"{tex_folder}gem.png")
gem_bad_id = load_texture(f"{tex_folder}gem_bad.png")
wood_id = load_texture(f"{tex_folder}wood.png")
key_id = load_texture(f"{tex_folder}key.png")
warp_id = load_texture(f"{tex_folder}warp.png")

env = GridEnv(
    template=GridTemplate.four_rooms_split, orientation_type=GridOrientation.fixed
)
env.reset()

acts = [3, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3]

img_list = []

timer = time.time()
for i in range(len(acts)):
    env.step(acts[i])

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    set_camera(env.agent_pos, env.looking)
    render_walls(env.blocks, env.agent_pos, env.looking)
    for reward in env.objects["rewards"]:
        if env.objects["rewards"][reward] > 0:
            render_sphere(reward[0], 0.0, reward[1], 0.25, texture=gem_id)
        else:
            render_sphere(reward[0], 0.0, reward[1], 0.25, texture=gem_bad_id)
    for door in env.objects["doors"]:
        render_cube(door[0], 0.0, door[1], wood_id)
    for key in env.objects["keys"]:
        render_sphere(key[0], -0.1, key[1], 0.1, texture=key_id)
    for warp in env.objects["warps"]:
        render_sphere(warp[0], -0.5, warp[1], 0.33, texture=warp_id)
    render_plane(5.0, -0.5, 5.0, 10.0, floor_id)

    buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image_np = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)[::-1]
    image_np = np.flip(image_np, axis=1)

    top_down = env.make_visual_obs()
    top_down = cv2.resize(top_down, (RESOLUTION * 2, RESOLUTION * 2))

    image_np = np.concatenate((image_np, top_down), axis=1)

    img_list.append(image_np)

print(time.time() - timer)

glfw.destroy_window(window)
glfw.terminate()

import imageio

# save a gif video of the agent's movement
imageio.mimsave("./out/test.gif", img_list, fps=2)
