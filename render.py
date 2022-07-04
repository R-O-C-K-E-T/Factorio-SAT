import argparse
import enum
import json
import math
import time
from contextlib import contextmanager
from typing import List, Optional

import ffmpeg
import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from pygame.locals import *

import blueprint
from direction import Direction
from tile import AssemblingMachine, BaseTile, EmptyTile, Inserter
import tilemaps
from solver import Belt, Splitter, UndergroundBelt
from util import *

BELT_ANIMATION_LENGTH = 16
SPLITTER_ANIMATION_LENGTH = 32


@contextmanager
def render_to_framebuffer(multisamples):
    _, _, width, height = glGetIntegerv(GL_VIEWPORT)

    renderbuffer = glGenRenderbuffers(1)
    glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer)
    if multisamples <= 1:
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height)
    else:
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, multisamples, GL_RGBA8, width, height)
    glBindRenderbuffer(GL_RENDERBUFFER, 0)

    framebuffer = int(glGenFramebuffers(1))
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer)

    try:
        yield framebuffer
    finally:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDeleteRenderbuffers(1, [renderbuffer])


def draw_texture(texture, width=None, height=None):
    assert width is not None or height is not None

    glBindTexture(GL_TEXTURE_2D, texture)
    texture_width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
    texture_height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)

    if width is None:
        width = height * (texture_width / texture_height)
    elif height is None:
        height = width * (texture_height / texture_width)

    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)

    glTexCoord2f(0, 0)
    glVertex2f(0, 0)

    glTexCoord2f(0, 1)
    glVertex2f(0, height)

    glTexCoord2f(1, 1)
    glVertex2f(width, height)

    glTexCoord2f(1, 0)
    glVertex2f(width, 0)

    glEnd()
    glDisable(GL_TEXTURE_2D)

    glBindTexture(GL_TEXTURE_2D, 0)


def get_animation_length(solution):
    for tile in solution.reshape(-1):
        if tile.get('is_splitter') is True:
            return SPLITTER_ANIMATION_LENGTH
    return BELT_ANIMATION_LENGTH


class RenderLayer(enum.Enum):
    BOTTOM = enum.auto()
    TOP = enum.auto()


def render_tile(tile: BaseTile, animation: int, layer: RenderLayer):
    if isinstance(tile, Belt):
        if layer == RenderLayer.BOTTOM:
            glPushMatrix()
            glTranslatef(-0.5, -0.5, 0)
            tilemaps.BELT.render(animation % 16, [[11, 8, 4, 7], [0, 2, 1, 3], [6, 10, 9, 5]]
                                 [(tile.output_direction.value - tile.input_direction.value + 1) % 4][tile.input_direction])
            glPopMatrix()
    elif isinstance(tile, UndergroundBelt):
        if layer == RenderLayer.BOTTOM:
            glPushMatrix()
            glTranslatef(-0.5, -0.5, 0)

            tilemaps.BELT.render(animation % 16, [[14, 12, 18, 16], [19, 17, 15, 13]][tile.is_input][tile.direction])
            glPopMatrix()
        elif layer == RenderLayer.TOP:
            glPushMatrix()
            glTranslatef(-1, -1, 0)
            tilemaps.UNDERGROUND.render([[3, 2, 1, 0], [1, 0, 3, 2]][tile.is_input][tile.direction], tile.is_input)

            glPopMatrix()
    elif isinstance(tile, Splitter):
        if layer == RenderLayer.BOTTOM:
            glPushMatrix()
            glTranslatef(-0.5, -0.5, 0)
            tilemaps.BELT.render(animation % 16, [0, 2, 1, 3][tile.direction])
            glPopMatrix()
        elif layer == RenderLayer.TOP:
            glPushMatrix()
            if tile.direction == Direction.RIGHT:
                if not tile.is_head:
                    glTranslatef(0, -0.5, 0)
                    tilemaps.SPLITTER_EAST[1].render(animation % 8, (animation // 8) % 4)
                else:
                    glTranslatef(0, -5/16, 0)
                    tilemaps.SPLITTER_EAST[0].render(animation % 8, (animation // 8) % 4)
            elif tile.direction == Direction.LEFT:
                if not tile.is_head:
                    glTranslatef(0, -5/16, 0)
                    tilemaps.SPLITTER_WEST[0].render(animation % 8, (animation // 8) % 4)
                else:
                    glTranslatef(0, -5/16, 0)
                    tilemaps.SPLITTER_WEST[1].render(animation % 8, (animation // 8) % 4)
            elif tile.direction == Direction.UP:
                if not tile.is_head:
                    tilemaps.SPLITTER_NORTH.render(animation % 8, (animation // 8) % 4, upper=(13/32, 1))
                else:
                    tilemaps.SPLITTER_NORTH.render(animation % 8, (animation // 8) % 4, lower=(13/32, 0))
            elif tile.direction == Direction.DOWN:
                glTranslatef(-4/32, 0, 0)
                if not tile.is_head:
                    tilemaps.SPLITTER_SOUTH.render(animation % 8, (animation // 8) % 4, lower=(14/32, 0))
                else:
                    glTranslatef(-1/32, 0, 0)
                    tilemaps.SPLITTER_SOUTH.render(animation % 8, (animation // 8) % 4, upper=(14/32, 1))
            else:
                assert False
            glPopMatrix()
    elif isinstance(tile, Inserter):
        if layer == RenderLayer.BOTTOM:
            glPushMatrix()
            glTranslatef(-(8 + 1.5)/32, (8 - (7.5 - 1))/32, 0)
            tilemaps.INSERTER_PLATFORM[tile.type].render((1 - tile.direction) % 4, 0)
            glPopMatrix()
        elif layer == RenderLayer.TOP:
            # Good enough
            t = (abs(1 - (animation / 16)) - 0.5)

            if tile.type == 0:  # Normal
                t *= 1.5
            elif tile.type == 1:  # Long
                t *= 3.75
            else:
                assert False

            target = np.array([0.5, 0.5]) + np.array(tile.direction.vec) * t
            centre = np.array([0.5, 0.5])
            delta = target - centre

            length = np.sqrt(np.sum(delta**2))

            interior_angle = math.degrees(math.asin(length / 2))
            if t < 0:
                interior_angle *= -1

            base_angle = 90 * (2 - tile.direction) + interior_angle

            glPushMatrix()

            glTranslatef(16/32, 16/32, 0)  # 3
            glRotatef(base_angle, 0, 0, 1)  # 2
            glTranslatef(-3.5/32, 0/32, 0)  # 1

            draw_texture(tilemaps.INSERTER_HAND_BASE[tile.type], height=1)

            glPopMatrix()

            glPushMatrix()

            glTranslatef(16/32, 16/32, 0)
            glRotatef(base_angle, 0, 0, 1)
            glTranslatef(-0/32, 29/32, 0)
            glRotatef(180 - (2 * interior_angle), 0, 0, 1)
            glTranslatef(-7/32, 0/32, 0)

            if animation > 16:
                draw_texture(tilemaps.INSERTER_HAND_CLOSED[tile.type], height=1)
            else:
                draw_texture(tilemaps.INSERTER_HAND_OPEN[tile.type], height=1)

            glPopMatrix()
    elif isinstance(tile, AssemblingMachine):
        if layer == RenderLayer.BOTTOM:
            glPushMatrix()

            if tile.x == 0:
                glTranslatef(-7.5/32, 0, 0)
                lower_x = 0
                upper_x = 12/32
            elif tile.x == 1:
                lower_x = 12/32
                upper_x = 21.6/32
            elif tile.x == 2:
                lower_x = 21.6/32
                upper_x = 1
            else:
                assert False
            if tile.y == 0:
                glTranslatef(0, -6.75/32, 0)
                lower_y = 21/32
                upper_y = 1
            elif tile.y == 1:
                lower_y = 11.75/32
                upper_y = 21/32
            elif tile.y == 2:
                lower_y = 0
                upper_y = 11.75/32
            else:
                assert False

            tilemaps.ASSEMBLING_MACHINE.render(animation % 8, (animation // 8) % 4, lower=(lower_x, lower_y), upper=(upper_x, upper_y))

            glPopMatrix()
    else:
        assert False


def render_solution(solution, animation: int, colouring=True, colour_count: Optional[int] = None, underground=True):
    if colouring:
        all_colours = set()
        for item in solution.reshape(-1):
            for key in ('colour', 'colour_ux', 'colour_uy'):
                colour = item.get(key, 0)
                if isinstance(colour, list):
                    all_colours.update(colour)
                else:
                    all_colours.add(colour)
        all_colours.discard(None)
        if colour_count is None:
            colour_count = len(all_colours)
            palette = create_palette(colour_count)
            palette = dict(zip(sorted(all_colours), palette))
        else:
            assert len(all_colours) <= colour_count
            palette = create_palette(colour_count)

    for layer in RenderLayer:
        for x in reversed(range(solution.shape[1])):
            for y in range(solution.shape[0]):
                item = solution[y, x]

                tile = blueprint.read_tile(item)
                if isinstance(tile, EmptyTile):
                    continue

                colour = item.get('colour', 0)
                if isinstance(colour, list):
                    colour = colour[0]

                glPushMatrix()
                glTranslatef(x, y, 0)

                if colouring:
                    if colour is None:
                        glColor3f(0.5, 0.5, 0.5)
                    else:
                        glColor3fv(palette[colour])
                else:
                    glColor3f(1, 1, 1)
                render_tile(tile, animation, layer)

                glPopMatrix()

    if underground:
        for x in reversed(range(solution.shape[0])):
            for y in range(solution.shape[1]):
                item = solution[x, y]

                try:
                    underground = item['underground']
                except KeyError:
                    continue

                glPushMatrix()

                glTranslatef(x, y, 0)
                if any(underground[0::2]):
                    colour = item.get('colour_ux', 0)
                    if isinstance(colour, list):
                        colour = colour[0]

                    if colouring:
                        if colour is None:
                            glColor3f(0.5, 0.5, 0.5)
                        else:
                            glColor3fv(palette[colour])
                    else:
                        glColor3f(1, 1, 1)
                    glBegin(GL_LINES)
                    glVertex2f(0, 0.5)
                    glVertex2f(1, 0.5)
                    glEnd()

                if any(underground[1::2]):
                    colour = item.get('colour_uy', 0)
                    if isinstance(colour, list):
                        colour = colour[0]

                    if colouring:
                        if colour is None:
                            glColor3f(0.5, 0.5, 0.5)
                        else:
                            glColor3fv(palette[colour])
                    else:
                        glColor3f(1, 1, 1)

                    glBegin(GL_LINES)
                    glVertex2f(0.5, 0)
                    glVertex2f(0.5, 1)
                    glEnd()
                glPopMatrix()


def render_attributes(solution, font, names: List[str]):
    if len(names) == 0:
        return

    texture = tilemaps.gen_texture_2d()

    glEnable(GL_TEXTURE_2D)
    glColor3f(1, 1, 1)

    text_grid = np.empty_like(solution, dtype=object)
    for y in range(solution.shape[0]):
        for x in range(solution.shape[1]):
            item = solution[y, x]

            lines = []
            for name in names:
                sub_item = item
                for piece in name.split('.'):
                    try:
                        sub_item = item[int(piece)]
                        continue
                    except ValueError:
                        pass

                    if piece not in sub_item:
                        raise RuntimeError('{} not found in tile {}'.format(name, item))
                    sub_item = sub_item[piece]

                text = str(np.array(sub_item))
                for line in text.split('\n'):
                    lines.append(line)

            text_grid[y, x] = lines

    max_width = 0
    max_height = 0
    for lines in text_grid.reshape(-1):
        sizes = [font.size(line) for line in lines]

        width = max(width for width, _ in sizes)
        height = sum(height for _, height in sizes)

        max_width = max(width, max_width)
        max_height = max(height, max_height)

    if max_width == 0 or max_height == 0:
        return

    scale = min(1 / max_width, 1 / max_height)

    for y in range(solution.shape[0]):
        for x in range(solution.shape[1]):
            lines = text_grid[y, x]
            glPushMatrix()
            glTranslatef(x, y, 0)

            for line in lines:
                surface = font.render(line, True, (255, 0, 0), (0, 0, 0))

                data = np.empty((*surface.get_size()[::-1], 4), dtype=np.uint8)
                data[:, :, -1] = pygame.surfarray.array3d(surface)[:, :, 0].T
                data[:, :, :-1] = 0, 0, 255

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surface.get_width(), surface.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

                tile_width = scale * surface.get_width()
                tile_height = scale * surface.get_height()

                glBegin(GL_QUADS)
                glTexCoord(0, 0)
                glVertex2f(0, 0)

                glTexCoord(0, 1)
                glVertex2f(0, tile_height)

                glTexCoord(1, 1)
                glVertex2f(tile_width, tile_height)

                glTexCoord(1, 0)
                glVertex2f(tile_width, 0)
                glEnd()

                glTranslatef(0, tile_height, 0)

            glPopMatrix()

    glDisable(GL_TEXTURE_2D)

    glDeleteTextures(1, [texture])


def render_grid(start_x: float, stop_x: float, start_y: float, stop_y: float):
    glBegin(GL_LINES)

    for x in range(math.floor(start_x), math.ceil(stop_x)):
        glVertex2f(x, start_y)
        glVertex2f(x, stop_y)

    for y in range(math.floor(start_y), math.ceil(stop_y)):
        glVertex2f(start_x, y)
        glVertex2f(stop_x, y)
    glEnd()


def draw_arrow():
    glBegin(GL_LINES)

    glVertex2f(0.1, 0.5)
    glVertex2f(0.9, 0.5)

    glVertex2f(0.6, 0.2)
    glVertex2f(0.9, 0.5)

    glVertex2f(0.6, 0.8)
    glVertex2f(0.9, 0.5)

    glEnd()


def HSVtoRGB(hue: float, sat: float, val: float):
    c = val * sat
    hue = (hue % 1) * 6
    x = c * (1 - abs((hue % 2) - 1))
    if hue > 5:
        out = c, 0, x
    elif hue > 4:
        out = x, 0, c
    elif hue > 3:
        out = 0, x, c
    elif hue > 2:
        out = 0, c, x
    elif hue > 1:
        out = x, c, 0
    else:
        out = c, x, 0
    return np.array(out) + (val - c)


def create_palette(colours):
    if colours == 1:
        return np.array([[1.0, 1.0, 1.0]])
    elif colours <= 3:
        return np.array([
            [1, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0.5, 1],
        ])[:colours]
    else:
        return np.array([HSVtoRGB(i / colours, 0.4, 1.0) for i in range(colours)])


def export_video(frames, filename):
    _, height, width, _ = frames.shape
    process = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))

    if filename.endswith('gif'):
        process = process.output(filename, loop=0, r=60)
    else:
        process = process.output(filename, pix_fmt='yuv420p', vcodec='libx264', r=60)

    process = process.overwrite_output().run_async(pipe_stdin=True)
    for frame in frames.astype(np.uint8):
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    process.wait()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders a grid of tiles from standard input')
    parser.add_argument('--hide-colour', action='store_true', help='Disables colouring the tiles based on their given colour')
    parser.add_argument('--show-underground', action='store_true', help='Shows the underground tile connections')
    parser.add_argument('--show-raw', action='append', help='Overlays then given tile attribute', default=[])
    parser.add_argument('--cell-size', type=int, default=32, help='Size of the grid cells (pixels)')
    parser.add_argument('--export-all', action='store_true', help='Renders all grids to videos then exits')
    parser.add_argument('--export-format', choices=['gif', 'video'], default='video', help='Format to export animations as')
    parser.add_argument('--padding', type=float, default=0, help='Amount of padding tiles around edges of grid')
    parser.add_argument('--colour-count', type=int, help='TODO')
    args = parser.parse_args()

    if args.cell_size <= 0:
        raise RuntimeError('Cell size must be greater than 0')

    padding_pixels = round(2 * args.padding * args.cell_size)

    def grid_size(solution):
        return max(solution.shape[1] * args.cell_size + padding_pixels, 1), max(solution.shape[0] * args.cell_size + padding_pixels, 1)

    def increment_solution(amount):
        global index, framebuffers, input_closed

        assert abs(amount) == 1
        index += amount
        while not input_closed and index >= len(solutions):
            try:
                solutions.append(np.array(json.loads(input()), ndmin=2))
            except EOFError:
                print('All solutions found')
                input_closed = True

        index = index % len(solutions)
        size = grid_size(solutions[index])
        pygame.display.set_mode(size, OPENGL | DOUBLEBUF)
        glViewport(0, 0, *size)

        glDeleteFramebuffers(len(framebuffers), list(framebuffers.values()))
        framebuffers = {}

    t = time.time()
    try:
        solution = np.array(json.loads(input()), ndmin=2)
    except EOFError:
        solution = None
    dt = time.time() - t
    print(dt)
    if solution is None:
        print('No solutions')
        exit()
    solutions = [solution]

    # print(solution.shape)

    pygame.display.init()
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    multisamples = 4
    '''
    if multisamples > 1:
        pygame.display.gl_set_attribute(GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES, multisamples)'''

    pygame.display.set_mode(grid_size(solutions[-1]), OPENGL | DOUBLEBUF)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    tilemaps.init()

    glClearColor(1, 1, 1, 1)

    clock = pygame.time.Clock()

    title = 'Belt Balancer'
    pygame.display.set_caption(title)

    input_closed = False

    waiting_to_save = False

    framebuffers = {}

    t = 0
    index = 0
    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN and not args.export_all:
                    if event.key == K_i:
                        increment_solution(1)
                    elif event.key == K_k:
                        increment_solution(-1)
                    elif event.key == K_s:
                        waiting_to_save = True
                    elif event.key == K_e:
                        tiles = solutions[index].copy()
                        for i, row in enumerate(tiles):
                            for j, entry in enumerate(row):
                                tiles[i, j] = blueprint.read_tile(entry)
                        print(blueprint.encode_blueprint(blueprint.make_blueprint(tiles)))
            pygame.display.set_caption('{} - {}: {:.2f}'.format(title, index, clock.get_fps()))

            animation_length = get_animation_length(solutions[index])
            animation = (t // 2) % animation_length
            if animation not in framebuffers:
                with render_to_framebuffer(multisamples) as framebuffer:
                    glClear(GL_COLOR_BUFFER_BIT)
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    gluOrtho2D(-args.padding, solutions[index].shape[1] + args.padding, solutions[index].shape[0] + args.padding, -args.padding)
                    glMatrixMode(GL_MODELVIEW)

                    glLineWidth(1)
                    glColor3f(1, 0, 0)
                    render_grid(-args.padding, solutions[index].shape[1] + args.padding, -args.padding, solutions[index].shape[0] + args.padding)

                    render_solution(solutions[index], animation, not args.hide_colour, args.colour_count, args.show_underground)
                    render_attributes(solutions[index], font, args.show_raw)

                framebuffers[animation] = framebuffer
            else:
                framebuffer = framebuffers[animation]

            _, _, width, height = glGetIntegerv(GL_VIEWPORT)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
            glDrawBuffer(GL_BACK)

            glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer)
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST)

            glBindFramebuffer(GL_FRAMEBUFFER, 0)

            if (waiting_to_save or args.export_all) and all(i in framebuffers for i in range(animation_length)):
                waiting_to_save = False

                texture = tilemaps.gen_texture_2d()
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

                intermediate_framebuffer = int(glGenFramebuffers(1))
                glBindFramebuffer(GL_FRAMEBUFFER, intermediate_framebuffer)
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)

                frames = np.empty((animation_length, height, width, 3), np.uint8)
                for i in range(animation_length):
                    framebuffer = framebuffers[i]
                    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer)
                    glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR)

                    data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE)
                    frames[i] = np.frombuffer(data, np.uint8).reshape((height, width, 3))

                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glBindTexture(GL_TEXTURE_2D, 0)

                glDeleteTextures(1, [texture])
                glDeleteFramebuffers(1, [intermediate_framebuffer])

                frames = frames[:, ::-1, :, :]  # Vertical flip

                if args.export_format == 'gif':
                    filename = '{}.gif'.format(index)
                else:
                    filename = '{}.mp4'.format(index)
                export_video(frames, filename)

                if args.export_all:
                    increment_solution(1)
                    if index == 0:  # Must have found all solutions
                        running = False

            pygame.display.flip()

            if not args.export_all:
                clock.tick(60)
            t += 1
    finally:
        pygame.quit()
