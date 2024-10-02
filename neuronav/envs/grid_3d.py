import glfw
import numpy as np
import os
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
    def __init__(self, resolution=128):
        self.resolution = resolution
        self.virtual_display = None
        self.initialize_glfw()
        self.configure_opengl()
        self.load_textures()
        self.display_lists = {}

    def initialize_glfw(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(
            self.resolution, self.resolution, "Offscreen", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        self.width, self.height = self.resolution, self.resolution

    def configure_opengl(self, fov=60.0):
        glViewport(0, 0, self.width, self.height)
        glClearColor(135 / 255, 206 / 255, 235 / 255, 1.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, self.width / self.height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def load_textures(self):
        self.tex_folder = os.path.join(os.path.dirname(__file__), "textures/")
        self.textures = {
            name: load_texture(f"{self.tex_folder}{file}")
            for name, file in {
                "floor": "floor.png",
                "wall": "wall.png",
                "gem": "gem.png",
                "gem_bad": "gem_bad.png",
                "wood": "wood.png",
                "key": "key.png",
                "warp": "warp.png",
            }.items()
        }

    def set_camera(self, agent_pos, agent_dir):
        offsets = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        offset = offsets[agent_dir]
        pos = (agent_pos[0], 0, agent_pos[1])
        target = (agent_pos[0] + offset[0], 0, agent_pos[1] + offset[1])
        up = (0, 1, 0)
        glLoadIdentity()
        gluLookAt(*pos, *target, *up)

    def create_walls_display_list(self, blocks):
        list_id = glGenLists(1)
        glNewList(list_id, GL_COMPILE)
        for block in blocks:
            render_cube(block[0], 0.0, block[1], self.textures["wall"])
        glEndList()
        return list_id

    def create_objects_display_list(self, env):
        list_id = glGenLists(1)
        glNewList(list_id, GL_COMPILE)
        for reward, info in env.objects["rewards"].items():
            reward_val = info[0] if isinstance(info, list) else info
            texture = (
                self.textures["gem"] if reward_val > 0 else self.textures["gem_bad"]
            )
            render_sphere(reward[0], 0.0, reward[1], 0.25, texture=texture)
        for door in env.objects["doors"]:
            render_cube(door[0], 0.0, door[1], self.textures["wood"])
        for key in env.objects["keys"]:
            render_sphere(key[0], -0.1, key[1], 0.1, texture=self.textures["key"])
        for warp in env.objects["warps"]:
            render_sphere(warp[0], -0.5, warp[1], 0.33, texture=self.textures["warp"])
        glEndList()
        return list_id

    def render_frame(self, env):
        glViewport(0, 0, self.width, self.height)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.set_camera(env.agent_pos, env.looking)

        # Render walls
        if "walls" not in self.display_lists or env.blocks != self.last_blocks:
            if "walls" in self.display_lists:
                glDeleteLists(self.display_lists["walls"], 1)
            self.display_lists["walls"] = self.create_walls_display_list(env.blocks)
            self.last_blocks = env.blocks.copy()
        glCallList(self.display_lists["walls"])

        # Render objects
        if "objects" not in self.display_lists or env.objects != self.last_objects:
            if "objects" in self.display_lists:
                glDeleteLists(self.display_lists["objects"], 1)
            self.display_lists["objects"] = self.create_objects_display_list(env)
            self.last_objects = env.objects.copy()
        glCallList(self.display_lists["objects"])

        # Render floor
        render_plane(
            env.grid_size / 2,
            -0.5,
            env.grid_size / 2,
            env.grid_size,
            self.textures["floor"],
        )

        # Read pixels
        buffer = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(
            self.height, self.width, 3
        )
        return np.flip(image, axis=0)  # Only flip vertically

    def close(self):
        for list_id in self.display_lists.values():
            glDeleteLists(list_id, 1)
        glfw.destroy_window(self.window)
        glfw.terminate()
        if self.virtual_display:
            self.virtual_display.stop()
