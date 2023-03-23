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
        self.textures["floor"] = load_texture(f"{self.tex_folder}floor.png")
        self.textures["wall"] = load_texture(f"{self.tex_folder}wall.png")
        self.textures["gem"] = load_texture(f"{self.tex_folder}gem.png")
        self.textures["gem_bad"] = load_texture(f"{self.tex_folder}gem_bad.png")
        self.textures["wood"] = load_texture(f"{self.tex_folder}wood.png")
        self.textures["key"] = load_texture(f"{self.tex_folder}key.png")
        self.textures["warp"] = load_texture(f"{self.tex_folder}warp.png")

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

    def render_objects(self, env):
        self.render_walls(env.blocks, env.agent_pos, env.looking)
        for reward in env.objects["rewards"]:
            reward_val = (
                env.objects["rewards"][reward][0]
                if isinstance(env.objects["rewards"][reward], list)
                else env.objects["rewards"][reward]
            )
            if reward_val > 0:
                render_sphere(
                    reward[0], 0.0, reward[1], 0.25, texture=self.textures["gem"]
                )
            else:
                render_sphere(
                    reward[0], 0.0, reward[1], 0.25, texture=self.textures["gem_bad"]
                )
        for door in env.objects["doors"]:
            render_cube(door[0], 0.0, door[1], self.textures["wood"])
        for key in env.objects["keys"]:
            render_sphere(key[0], -0.1, key[1], 0.1, texture=self.textures["key"])
        for warp in env.objects["warps"]:
            render_sphere(warp[0], -0.5, warp[1], 0.33, texture=self.textures["warp"])

    def render_walls(self, blocks, agent_pos, agent_dir):
        for block in blocks:
            # check if the block is in the agent's field of view
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
            render_cube(block[0], 0.0, block[1], self.textures["wall"])

    def render_frame(self, env):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self.set_camera(env.agent_pos, env.looking)
        self.render_walls(env.blocks, env.agent_pos, env.looking)
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
