import glfw
import numpy as np
import os
import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from pyvirtualdisplay import Display
from neuronav.envs.gl_utils import (
    load_texture,
    render_plane,
    render_cube,
    render_sphere,
)


class Grid3DRenderer:
    def __init__(self, resolution=256):
        self.virtual_display = None
        self.initialize_glfw(resolution)
        self.configure_opengl()
        self.width, self.height = glfw.get_framebuffer_size(self.window)
        self.tex_folder = os.path.join(os.path.dirname(__file__), "textures/")
        self.textures = {}
        self.load_textures()
        self.cached_walls = None
        self.cached_objects = None

    def load_textures(self):
        texture_files = {
            "floor": "floor.png",
            "wall": "wall.png",
            "gem": "gem.png",
            "gem_bad": "gem_bad.png",
            "wood": "wood.png",
            "key": "key.png",
            "warp": "warp.png",
        }
        for name, file in texture_files.items():
            self.textures[name] = load_texture(f"{self.tex_folder}{file}")

    def initialize_display(self):
        if sys.platform != "win32":
            self.virtual_display = Display(visible=0, size=(1, 1))
            self.virtual_display.start()

    def initialize_glfw(self, resolution):
        try:
            glfw.init()
        except:
            self.initialize_display()
            glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(resolution, resolution, "Offscreen", None, None)
        # Adjust the window size based on the content scale
        x_scale, y_scale = glfw.get_window_content_scale(window)
        adjusted_width = int(resolution / x_scale)
        adjusted_height = int(resolution / y_scale)
        glfw.set_window_size(window, adjusted_width, adjusted_height)
        glfw.make_context_current(window)
        self.window = window

    def configure_opengl(self, fov=60.0):
        glClearColor(135 / 255, 206 / 255, 235 / 255, 1.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, 1, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def set_camera(self, agent_pos, agent_dir):
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

    def render_walls(self, blocks):
        if self.cached_walls is None or blocks != self.cached_walls:
            self.cached_walls = blocks
            glNewList(1, GL_COMPILE)
            for block in blocks:
                render_cube(block[0], 0.0, block[1], self.textures["wall"])
            glEndList()
        glCallList(1)

    def render_objects(self, env):
        if self.cached_objects != env.objects:
            self.cached_objects = env.objects.copy()
            glNewList(2, GL_COMPILE)
            for reward in env.objects["rewards"]:
                reward_val = (
                    env.objects["rewards"][reward][0]
                    if isinstance(env.objects["rewards"][reward], list)
                    else env.objects["rewards"][reward]
                )
                texture = (
                    self.textures["gem"] if reward_val > 0 else self.textures["gem_bad"]
                )
                render_sphere(reward[0], 0.0, reward[1], 0.25, texture=texture)
            for door in env.objects["doors"]:
                render_cube(door[0], 0.0, door[1], self.textures["wood"])
            for key in env.objects["keys"]:
                render_sphere(key[0], -0.1, key[1], 0.1, texture=self.textures["key"])
            for warp in env.objects["warps"]:
                render_sphere(
                    warp[0], -0.5, warp[1], 0.33, texture=self.textures["warp"]
                )
            glEndList()
        glCallList(2)

    def render_frame(self, env):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.set_camera(env.agent_pos, env.looking)
        self.render_walls(env.blocks)
        self.render_objects(env)
        render_plane(
            env.grid_size / 2,
            -0.5,
            env.grid_size / 2,
            env.grid_size,
            self.textures["floor"],
        )
        buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image_np = np.frombuffer(buffer, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )[::-1]
        image_np = np.flip(image_np, axis=1)
        return image_np

    def close(self):
        if self.virtual_display is not None:
            self.virtual_display.stop()
        glfw.destroy_window(self.window)
        glfw.terminate()
