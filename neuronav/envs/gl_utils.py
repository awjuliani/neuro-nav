import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import math


def load_texture(filename):
    img = Image.open(filename).convert("RGB")
    img_data = np.array(img, dtype=np.uint8)

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

    half_area = area / 2.0

    glPushMatrix()
    glTranslatef(x, y, z)
    glBegin(GL_QUADS)
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
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-0.5, 0.5, -0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(0.5, 0.5, -0.5)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(0.5, -0.5, -0.5)

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
    glVertex3f(0.5, -0.5, -0.5)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(0.5, -0.5, 0.5)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-0.5, -0.5, 0.5)

    # Right face
    glTexCoord2f(1.0, 0.0)
    glVertex3f(0.5, -0.5, -0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(0.5, 0.5, -0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(0.5, 0.5, 0.5)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(0.5, -0.5, 0.5)

    # Left face
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-0.5, -0.5, -0.5)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-0.5, -0.5, 0.5)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-0.5, 0.5, 0.5)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-0.5, 0.5, -0.5)

    glEnd()
    glDisable(GL_TEXTURE_2D)
    glPopMatrix()


def render_sphere(x, y, z, radius, slices=16, stacks=16, texture=None):
    glPushMatrix()
    glTranslatef(x, y, z)

    if texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)

    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = math.sin(lat0)
        zr0 = math.cos(lat0)

        lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
        z1 = math.sin(lat1)
        zr1 = math.cos(lat1)

        glBegin(GL_QUAD_STRIP)
        for j in range(slices + 1):
            lng = 2 * math.pi * float(j) / slices
            x = math.cos(lng)
            y = math.sin(lng)

            glNormal3f(x * zr0, y * zr0, z0)
            glTexCoord2f(float(j) / slices, float(i) / stacks)
            glVertex3f(x * zr0 * radius, y * zr0 * radius, z0 * radius)

            glNormal3f(x * zr1, y * zr1, z1)
            glTexCoord2f(float(j) / slices, float(i + 1) / stacks)
            glVertex3f(x * zr1 * radius, y * zr1 * radius, z1 * radius)
        glEnd()

    if texture is not None:
        glDisable(GL_TEXTURE_2D)

    glPopMatrix()
