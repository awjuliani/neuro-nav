import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import math

# Cache for loaded textures
texture_cache = {}


def load_texture(filename):
    if filename in texture_cache:
        return texture_cache[filename]

    img = Image.open(filename).convert("RGB")
    img_data = np.array(img, dtype=np.uint8)

    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
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
    glGenerateMipmap(GL_TEXTURE_2D)

    texture_cache[filename] = texture_id
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

    vertices = [
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (0.5, 0.5, -0.5),
        (0.5, -0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, -0.5),
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5),
        (0.5, -0.5, 0.5),
        (-0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, 0.5, -0.5),
    ]

    tex_coords = [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 1),
        (0, 0),
        (0, 1),
        (0, 0),
        (1, 0),
        (1, 1),
        (1, 1),
        (0, 1),
        (0, 0),
        (1, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (0, 0),
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
    ]

    glBegin(GL_QUADS)
    for i in range(0, len(vertices), 4):
        for j in range(4):
            glTexCoord2f(*tex_coords[i + j])
            glVertex3f(*vertices[i + j])
    glEnd()

    glDisable(GL_TEXTURE_2D)
    glPopMatrix()


def render_sphere(x, y, z, radius, slices=16, stacks=16, texture=None):
    glPushMatrix()
    glTranslatef(x, y, z)

    if texture is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)

    quadric = gluNewQuadric()
    gluQuadricTexture(quadric, GL_TRUE)
    gluSphere(quadric, radius, slices, stacks)
    gluDeleteQuadric(quadric)

    if texture is not None:
        glDisable(GL_TEXTURE_2D)

    glPopMatrix()
