import os

from typing import *
from os import path

from OpenGL.GL import *
from PIL import Image

ASSETS_DIR = os.path.join(os.getenv("XDG_DATA_HOME"), "factorio-sat/assets")

def get_texture_size(texture: int) -> Tuple[int, int]:
    glBindTexture(GL_TEXTURE_2D, texture)
    width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
    height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)

    return width, height


def gen_texture_2d():
    texture = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    return texture


def load_image(filename, texture=None):
    if texture is None:
        texture = gen_texture_2d()

    img = Image.open(filename)
    img = img.transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA')
    data = img.tobytes()

    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *img.size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    return texture


class Tilemap:
    def __init__(self, texture: int, entry_size: Tuple[int, int], pixels_per_unit: int):
        self.texture = texture
        self.entry_size = entry_size
        self.pixels_per_unit = pixels_per_unit
        self.texture_size = get_texture_size(texture)

        assert self.texture_size[0] % entry_size[0] == 0
        assert self.texture_size[1] % entry_size[1] == 0

    def render(self, x, y, lower=(0, 0), upper=(1, 1)):
        glPushMatrix()

        glScalef(
            (upper[0] - lower[0]) * (self.entry_size[0] / self.pixels_per_unit),
            (upper[1] - lower[1]) * (self.entry_size[1] / self.pixels_per_unit),
            1
        )

        glBindTexture(GL_TEXTURE_2D, self.texture)

        y = self.texture_size[1] // self.entry_size[1] - y - 1

        glEnable(GL_TEXTURE_2D)

        min_x = (x + lower[0]) * (self.entry_size[0] / self.texture_size[0])
        min_y = (y + lower[1]) * (self.entry_size[1] / self.texture_size[1])
        max_x = (x + upper[0]) * (self.entry_size[0] / self.texture_size[0])
        max_y = (y + upper[1]) * (self.entry_size[1] / self.texture_size[1])

        glBegin(GL_QUADS)

        glTexCoord2f(min_x, max_y)
        glVertex2f(0, 0)

        glTexCoord2f(min_x, min_y)
        glVertex2f(0, 1)

        glTexCoord2f(max_x, min_y)
        glVertex2f(1, 1)

        glTexCoord2f(max_x, max_y)
        glVertex2f(1, 0)

        glEnd()

        glDisable(GL_TEXTURE_2D)
        glPopMatrix()


def init():
    global BELT, UNDERGROUND, SPLITTER_EAST, SPLITTER_WEST, SPLITTER_NORTH, SPLITTER_SOUTH, INSERTER_PLATFORM, INSERTER_HAND_BASE, INSERTER_HAND_OPEN, INSERTER_HAND_CLOSED, ASSEMBLING_MACHINE

    BELT = Tilemap(load_image(path.join(ASSETS_DIR, 'hr-transport-belt.png')), (128, 128), PIXELS_PER_UNIT)
    UNDERGROUND = Tilemap(load_image(path.join(ASSETS_DIR, 'hr-underground-belt-structure.png')), (192, 192), PIXELS_PER_UNIT)
    SPLITTER_EAST = [
        Tilemap(load_image(path.join(ASSETS_DIR, 'hr-splitter-east.png')), (90, 84), PIXELS_PER_UNIT),
        Tilemap(load_image(path.join(ASSETS_DIR, 'hr-splitter-east-top_patch.png')), (90, 104), PIXELS_PER_UNIT),
    ]
    SPLITTER_WEST = [
        Tilemap(load_image(path.join(ASSETS_DIR, 'hr-splitter-west.png')), (90, 86), PIXELS_PER_UNIT),
        Tilemap(load_image(path.join(ASSETS_DIR, 'hr-splitter-west-top_patch.png')), (90, 96), PIXELS_PER_UNIT),
    ]
    SPLITTER_SOUTH = Tilemap(load_image(path.join(ASSETS_DIR, 'hr-splitter-south.png')), (164, 64), PIXELS_PER_UNIT)
    SPLITTER_NORTH = Tilemap(load_image(path.join(ASSETS_DIR, 'hr-splitter-north.png')), (160, 70), PIXELS_PER_UNIT)

    INSERTER_PLATFORM = Tilemap(load_image(path.join(ASSETS_DIR, 'hr-inserter-platform.png')), (105, 79),
                                PIXELS_PER_UNIT), Tilemap(load_image(path.join(ASSETS_DIR, 'hr-long-handed-inserter-platform.png')), (105, 79), PIXELS_PER_UNIT)

    INSERTER_HAND_BASE = load_image(path.join(ASSETS_DIR, 'hr-inserter-hand-base.png')), load_image(path.join(ASSETS_DIR, 'hr-long-handed-inserter-hand-base.png'))
    INSERTER_HAND_OPEN = load_image(path.join(ASSETS_DIR, 'hr-inserter-hand-open.png')), load_image(path.join(ASSETS_DIR, 'hr-long-handed-inserter-hand-open.png'))
    INSERTER_HAND_CLOSED = load_image(path.join(ASSETS_DIR, 'hr-inserter-hand-closed.png')), load_image(path.join(ASSETS_DIR, 'hr-long-handed-inserter-hand-closed.png'))

    ASSEMBLING_MACHINE = Tilemap(load_image(path.join(ASSETS_DIR, 'hr-assembling-machine-1.png')), (214, 226), PIXELS_PER_UNIT)


PIXELS_PER_UNIT = 64

BELT = None
UNDERGROUND = None
SPLITTER_EAST = None
SPLITTER_WEST = None
SPLITTER_NORTH = None
SPLITTER_SOUTH = None
INSERTER_PLATFORM = None
INSERTER_HAND_BASE = None
INSERTER_HAND_OPEN = None
INSERTER_HAND_CLOSED = None
ASSEMBLING_MACHINE = None
