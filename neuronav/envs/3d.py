import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import time

from neuronav.envs.grid_env import GridEnv, GridOrientation, GridTemplate


def load_texture(filename):
    img = Image.open(filename)
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


def render_plane(x, y, z, area, texture_id):
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    glPushMatrix()
    glTranslatef(x, y, z)
    glBegin(GL_QUADS)

    half_area = area / 2.0

    # Plane
    glTexCoord2f(0, 0)
    glVertex3f(-half_area, 0.0, half_area)
    glTexCoord2f(1, 0)
    glVertex3f(half_area, 0.0, half_area)
    glTexCoord2f(1, 1)
    glVertex3f(half_area, 0.0, -half_area)
    glTexCoord2f(0, 1)
    glVertex3f(-half_area, 0.0, -half_area)

    glEnd()
    glPopMatrix()

    glDisable(GL_TEXTURE_2D)


def render_cube(x, y, z):
    glPushMatrix()
    glTranslatef(x, y, z)
    glBegin(GL_QUADS)

    # Front face
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)

    # Back face
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)

    # Left face
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, -0.5)

    # Right face
    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)

    # Top face
    glColor3f(0.0, 1.0, 1.0)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, -0.5)

    # Bottom face
    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)

    glEnd()
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


# Initialize GLFW for off-screen rendering
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(128, 128, "Offscreen", None, None)
glfw.make_context_current(window)

glClearColor(0.0, 0.5, 1.0, 1.0)
glEnable(GL_DEPTH_TEST)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45.0, 800 / 600, 0.1, 100.0)
glMatrixMode(GL_MODELVIEW)

width, height = glfw.get_framebuffer_size(window)
texture_id = load_texture("floor_tex.jpg")

env = GridEnv(template=GridTemplate.obstacle, orientation_type=GridOrientation.variable)
env.reset()

# test how long it takes to render 100 frames
timer = time.time()
for i in range(5):
    env.step(0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    set_camera(env.agent_pos, env.looking)

    for block in env.blocks:
        render_cube(block[0], 0.0, block[1])

    render_plane(5.0, -0.5, 5.0, 10.0, texture_id)

    buffer = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image_np = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)[::-1]
    image = Image.fromarray(image_np)
    image.save(f"test_{i}.png")

print(time.time() - timer)

glfw.destroy_window(window)
glfw.terminate()
