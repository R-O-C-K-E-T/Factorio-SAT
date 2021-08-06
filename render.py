import pygame, time, math, random, json, argparse, os
from contextlib import contextmanager

import ffmpeg
from PIL import Image

import numpy as np
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *


from solver import ALL_TILES, Belt, UndergroundBelt, Splitter
from util import *
import blueprint

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

def gen_texture_2d():
    texture = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    return texture

def draw_texture(texture, width=None, height=None):
    assert width is not None or height is not None

    glBindTexture(GL_TEXTURE_2D, texture)
    texture_width  = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
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

def load_image(filename, texture=None):
    if texture is None:
        texture = gen_texture_2d()

    img = Image.open(filename)
    img = img.transpose(Image.FLIP_TOP_BOTTOM).convert('RGBA')
    data = img.tobytes()
    
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, *img.size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
    return texture

def get_texture_size(texture):
    glBindTexture(GL_TEXTURE_2D, texture)
    width = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
    height = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
    
    return width, height

class Tilemap:
    def __init__(self, texture, entry_size, pixels_per_unit):
        self.texture = texture
        self.entry_size = entry_size
        self.pixels_per_unit = pixels_per_unit
        self.texture_size = get_texture_size(texture)

        assert self.texture_size[0] % entry_size[0] == 0
        assert self.texture_size[1] % entry_size[1] == 0

    def render(self, x, y, lower=(0,0), upper=(1,1)):
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

'''def render_tile_cached(tile, animation, layer):
    displaylist = TILE_DISPLAYLISTS[layer][tile][animation]
    glCallList(displaylist)'''

def get_animation_length(solution):
    for tile in solution.reshape(-1):
        if tile.get('is_splitter') is not None:
            return SPLITTER_ANIMATION_LENGTH
    return BELT_ANIMATION_LENGTH

def render_tile(tile, animation, layer):
    if isinstance(tile, Belt):
        if layer == 0:
            glPushMatrix()
            glTranslatef(-0.5, -0.5, 0)
            BELT_TILEMAP.render(animation % 16, [[11,8,4,7], [0,2,1,3], [6,10,9,5]][(tile.output_direction - tile.input_direction + 1) % 4][tile.input_direction])
            glPopMatrix()
    elif isinstance(tile, UndergroundBelt):
        if layer == 0:
            glPushMatrix()
            glTranslatef(-0.5, -0.5, 0)
            
            BELT_TILEMAP.render(animation % 16, [[14,12,18,16], [19,17,15,13]][tile.is_input][tile.direction])
            glPopMatrix()
        else:
            glPushMatrix()
            glTranslatef(-1, -1, 0)
            UNDERGROUND_TILEMAP.render([[3,2,1,0], [1,0,3,2]][tile.is_input][tile.direction], tile.is_input)
            
            glPopMatrix()
    elif isinstance(tile, Splitter):
        if layer == 0:
            glPushMatrix()
            glTranslatef(-0.5, -0.5, 0)
            BELT_TILEMAP.render(animation % 16, [0,2,1,3][tile.direction])
            glPopMatrix()
        else:
            glPushMatrix()
            if tile.direction == 0:
                if tile.side:
                    glTranslatef(0, -0.5, 0)
                    SPLITTER_EAST_TILEMAPS[1].render(animation % 8, (animation // 8) % 4)
                else:
                    glTranslatef(0, -5/16, 0)
                    SPLITTER_EAST_TILEMAPS[0].render(animation % 8, (animation // 8) % 4)
            elif tile.direction == 2:
                if tile.side:
                    glTranslatef(0, -5/16, 0)
                    SPLITTER_WEST_TILEMAPS[0].render(animation % 8, (animation // 8) % 4)
                else:
                    glTranslatef(0, -5/16, 0)
                    SPLITTER_WEST_TILEMAPS[1].render(animation % 8, (animation // 8) % 4)
            elif tile.direction == 1:
                if tile.side:
                    SPLITTER_NORTH_TILEMAP.render(animation % 8, (animation // 8) % 4, upper=(13/32,1))
                else:
                    SPLITTER_NORTH_TILEMAP.render(animation % 8, (animation // 8) % 4, lower=(13/32,0))
            elif tile.direction == 3:
                glTranslatef(-4/32, 0, 0)
                if tile.side:
                    SPLITTER_SOUTH_TILEMAP.render(animation % 8, (animation // 8) % 4, lower=(14/32,0))
                else:
                    glTranslatef(-1/32, 0, 0)
                    SPLITTER_SOUTH_TILEMAP.render(animation % 8, (animation // 8) % 4, upper=(14/32,1))
            glPopMatrix()
    elif isinstance(tile, Inserter):
        if layer == 0:
            glPushMatrix()
            glTranslatef(-(8 + 1.5)/32, (8 - (7.5 - 1))/32, 0)
            INSERTER_PLATFORM_TILEMAP[tile.type].render((1 - tile.direction) % 4, 0)
            glPopMatrix()
        elif layer == 1:
            # Good enough
            t = (abs(1 - (animation / 16)) - 0.5)

            if tile.type == 0: # Normal
                t *= 1.5
            elif tile.type == 1: # Long
                t *= 3.75
            else:
                assert False

            target = np.array([0.5, 0.5]) + np.array(direction_to_vec(tile.direction)) * t
            centre = np.array([0.5, 0.5])
            delta = target - centre

            length = np.sqrt(np.sum(delta**2))

            interior_angle = math.degrees(math.asin(length / 2))
            if t < 0:
                interior_angle *= -1


            base_angle = 90 * (2 - tile.direction) + interior_angle

            glPushMatrix()

            glTranslatef(16/32, 16/32, 0) # 3
            glRotatef(base_angle, 0, 0, 1) # 2
            glTranslatef(-3.5/32, 0/32, 0) # 1

            draw_texture(INSERTER_HAND_BASE[tile.type], height=1)

            glPopMatrix()

            glPushMatrix()

            glTranslatef(16/32, 16/32, 0)
            glRotatef(base_angle, 0, 0, 1)
            glTranslatef(-0/32, 29/32, 0)
            glRotatef(180 - (2 * interior_angle), 0, 0, 1)
            glTranslatef(-7/32, 0/32, 0)

            if animation > 16:
                draw_texture(INSERTER_HAND_CLOSED[tile.type], height=1)
            else:
                draw_texture(INSERTER_HAND_OPEN[tile.type], height=1)
            
            glPopMatrix()
    elif isinstance(tile, AssemblingMachine):
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

        ASSEMBLING_MACHINE_TILEMAP.render(animation % 8, (animation // 8) % 4, lower=(lower_x, lower_y), upper=(upper_x, upper_y))

        glPopMatrix()
    else:
        assert False

def render_solution(solution, animation, colouring=True, underground=True):
    all_colours = set()
    for item in solution.reshape(-1):
        for key in ('colour', 'colour_ux', 'colour_uy'):
            colour = item.get(key, 0)
            if isinstance(colour, list):
                all_colours.update(colour)
            else:
                all_colours.add(colour)
    all_colours.discard(None)

    if colouring:
        palette = create_palette(len(all_colours))
    else:
        palette = np.ones((len(all_colours), 3))
    palette = dict(zip(sorted(all_colours), palette))            

    for layer in range(2):
        for x in reversed(range(solution.shape[0])):
            for y in range(solution.shape[1]):
                item = solution[x, y]
                
                tile = blueprint.read_tile(item)
                if tile is None:
                    continue

                colour = item.get('colour', 0)
                if isinstance(colour, list):
                    colour = colour[0]

                
                        
                glPushMatrix()
                glTranslatef(x, y, 0)
                
                glColor3fv(palette[colour])
                render_tile(tile, animation, layer)
                #render_tile_cached(tile, animation, layer)
                
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
                    
                    glColor3fv(palette[colour])
                    glBegin(GL_LINES)
                    glVertex2f(0, 0.5)
                    glVertex2f(1, 0.5)
                    glEnd()

                if any(underground[1::2]):
                    colour = item.get('colour_uy', 0)
                    if isinstance(colour, list):
                        colour = colour[0]
                    glColor3fv(palette[colour])
                    glBegin(GL_LINES)
                    glVertex2f(0.5, 0)
                    glVertex2f(0.5, 1)
                    glEnd() 
                glPopMatrix()


def render_attributes(solution, names):
    if len(names) == 0:
        return

    texture = gen_texture_2d()

    glEnable(GL_TEXTURE_2D)
    glColor3f(1,1,1)

    for x in range(solution.shape[0]):
        for y in range(solution.shape[1]):
            item = solution[x, y]

            lines = []
            for name in names:
                if name not in item:
                    raise RuntimeError('{} not found in tile {}'.format(name, item))
                '''def stringify(value):
                    if value is None:
                        return '-'
                    elif isinstance(value, bool):
                        if value:
                            return 'P'
                        else:
                            return 'N'
                    elif isinstance(value, int):
                        return str(value)
                    else:
                        if len(value) == 0:
                            return '[]'
                        elif isinstance(value[0], int):
                            return '[' + ','.join(stringify(item) for item in value) + ']'
                        else:
                            return '[' + ''.join(stringify(item) for item in value) + ']'
                text = stringify(item[name])'''
                text = str(np.array(item[name]))
                for line in text.split('\n'):
                    surface = FONT.render(line, True, (255, 0, 0), (0,0,0))
                    lines.append(surface)

            width = max(surface.get_width() for surface in lines)
            height = sum(surface.get_height() for surface in lines)

            if width == 0:
                continue

            scale = min(1 / width, 1 / height)

            glPushMatrix()
            glTranslatef(x, y, 0)
            
            for surface in lines:
                data = np.empty((*surface.get_size()[::-1], 4), dtype=np.uint8)
                data[:,:,-1] = pygame.surfarray.array3d(surface)[:,:,0].T
                data[:,:,:-1] = 0, 0, 255

                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surface.get_width(), surface.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, data)

                tile_width  = scale * surface.get_width()
                tile_height = scale * surface.get_height()

                

                glBegin(GL_QUADS)
                glTexCoord(0,0)
                glVertex2f(0,0)
                
                glTexCoord(0,1)
                glVertex2f(0,tile_height)
                
                glTexCoord(1,1)
                glVertex2f(tile_width,tile_height)
                
                glTexCoord(1,0)
                glVertex2f(tile_width,0)
                glEnd()

                glTranslatef(0, tile_height, 0)

            glPopMatrix()

    glDisable(GL_TEXTURE_2D)            

    glDeleteTextures(1, [texture])



def render_grid(width, height):
    glBegin(GL_LINES)
            
    for x in range(1, width):
        glVertex2f(x, 0)
        glVertex2f(x, height)
                
    for y in range(1, height):
        glVertex2f(0, y)
        glVertex2f(width, y)
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

def HSVtoRGB(hue, sat, val):
    c = val * sat
    hue = (hue % 1) * 6
    x = c * (1 - abs((hue % 2) - 1))
    if   hue > 5:
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
        return np.array([[1.0,1.0,1.0]])
    elif colours <= 3:
        return np.array([
            [1, 0.5, 0.5],
            [0.5, 1, 0.5],
            [0.5, 0.5, 1],
        ])[:colours]
    else:
        return np.array([HSVtoRGB(i / colours, 0.4, 1.0) for i in range(colours)])

def init_tilemaps():
    global BELT_TILEMAP, UNDERGROUND_TILEMAP, SPLITTER_EAST_TILEMAPS, SPLITTER_WEST_TILEMAPS, SPLITTER_NORTH_TILEMAP, SPLITTER_SOUTH_TILEMAP, INSERTER_PLATFORM_TILEMAP, INSERTER_HAND_BASE, INSERTER_HAND_OPEN, INSERTER_HAND_CLOSED, ASSEMBLING_MACHINE_TILEMAP, TILE_DISPLAYLISTS, FONT
    PIXELS_PER_UNIT = 64

    BELT_TILEMAP = Tilemap(load_image('assets/hr-transport-belt.png'), (128, 128), PIXELS_PER_UNIT)
    UNDERGROUND_TILEMAP = Tilemap(load_image('assets/hr-underground-belt-structure.png'), (192, 192), PIXELS_PER_UNIT)
    SPLITTER_EAST_TILEMAPS = [
        Tilemap(load_image('assets/hr-splitter-east.png'), (90, 84), PIXELS_PER_UNIT),
        Tilemap(load_image('assets/hr-splitter-east-top_patch.png'), (90, 104), PIXELS_PER_UNIT),
    ]
    SPLITTER_WEST_TILEMAPS = [
        Tilemap(load_image('assets/hr-splitter-west.png'), (90, 86), PIXELS_PER_UNIT),
        Tilemap(load_image('assets/hr-splitter-west-top_patch.png'), (90, 96), PIXELS_PER_UNIT),
    ]
    SPLITTER_SOUTH_TILEMAP = Tilemap(load_image('assets/hr-splitter-south.png'), (164, 64), PIXELS_PER_UNIT)
    SPLITTER_NORTH_TILEMAP = Tilemap(load_image('assets/hr-splitter-north.png'), (160, 70), PIXELS_PER_UNIT)

    INSERTER_PLATFORM_TILEMAP = Tilemap(load_image('assets/hr-inserter-platform.png'), (105, 79), PIXELS_PER_UNIT), Tilemap(load_image('assets/hr-long-handed-inserter-platform.png'), (105, 79), PIXELS_PER_UNIT)


    INSERTER_HAND_BASE = load_image('assets/hr-inserter-hand-base.png'), load_image('assets/hr-long-handed-inserter-hand-base.png')
    INSERTER_HAND_OPEN = load_image('assets/hr-inserter-hand-open.png'), load_image('assets/hr-long-handed-inserter-hand-open.png')
    INSERTER_HAND_CLOSED = load_image('assets/hr-inserter-hand-closed.png'), load_image('assets/hr-long-handed-inserter-hand-closed.png')

    ASSEMBLING_MACHINE_TILEMAP = Tilemap(load_image('assets/hr-assembling-machine-1.png'), (214, 226), PIXELS_PER_UNIT)

    '''TILE_DISPLAYLISTS = [{}, {}]
    for layer in range(2):
        for tile in ALL_TILES[1:]:
            displaylists = [int(glGenLists(1)) for _ in range(ANIMATION_LENGTH)]
            for animation, displaylist in enumerate(displaylists):
                glNewList(displaylist, GL_COMPILE)
                render_tile(tile, animation, layer)
                glEndList()
            TILE_DISPLAYLISTS[layer][tile] = displaylists'''

    pygame.font.init()
    FONT = pygame.font.Font(None, 24)

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

BELT_TILEMAP = None
UNDERGROUND_TILEMAP = None
SPLITTER_EAST_TILEMAPS = None
SPLITTER_WEST_TILEMAPS = None
SPLITTER_NORTH_TILEMAP = None
SPLITTER_SOUTH_TILEMAP = None
INSERTER_PLATFORM_TILEMAP = None
INSERTER_HAND_BASE = None
INSERTER_HAND_OPEN = None
INSERTER_HAND_CLOSED = None
ASSEMBLING_MACHINE_TILEMAP = None
FONT = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders a grid of tiles from standard input')
    parser.add_argument('--hide-colour', action='store_true', help='Disables colouring the tiles based on their given colour')
    parser.add_argument('--show-underground', action='store_true', help='Shows the underground tile connections')
    parser.add_argument('--show-raw', action='append', help='Overlays then given tile attribute', default=[])
    parser.add_argument('--cell-size', type=int, default=32, help='Size of the grid cells (pixels)')
    parser.add_argument('--export-all', action='store_true', help='Renders all grids to videos then exits')
    parser.add_argument('--export-format', choices=['gif', 'video'], default='video', help='Format to export animations as')
    args = parser.parse_args()

    if args.cell_size <= 0:
        raise RuntimeError('Cell size must be greater than 0')

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
        pygame.display.set_mode((max(solutions[index].shape[0] * GRID_CELL_SIZE, 1), max(solutions[index].shape[1] * GRID_CELL_SIZE, 1)), OPENGL|DOUBLEBUF)
        glViewport(0, 0, solutions[index].shape[0] * GRID_CELL_SIZE, solutions[index].shape[1] * GRID_CELL_SIZE)

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

    #print(solution.shape)

    pygame.display.init()
    pygame.font.init()
    multisamples = 4
    '''
    if multisamples > 1:
        pygame.display.gl_set_attribute(GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES, multisamples)'''

    GRID_CELL_SIZE = args.cell_size
    pygame.display.set_mode((max(solutions[-1].shape[0] * GRID_CELL_SIZE, 1), max(solutions[-1].shape[1] * GRID_CELL_SIZE, 1)), OPENGL|DOUBLEBUF)
    
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    init_tilemaps()
    
    glClearColor(1,1,1,1)

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
                    gluOrtho2D(0, solutions[index].shape[0], solutions[index].shape[1], 0)
                    glMatrixMode(GL_MODELVIEW)

                    glLineWidth(1)
                    glColor3f(1,0,0)
                    render_grid(solutions[index].shape[0], solutions[index].shape[1])
                    
                    render_solution(solutions[index], animation, not args.hide_colour, args.show_underground)
                    render_attributes(solutions[index], args.show_raw)
                
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

                texture = gen_texture_2d()
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

                frames = frames[:, ::-1, :, :] # Vertical flip
                
                if args.export_format == 'gif':
                    filename = '{}.gif'.format(index)
                else:
                    filename = '{}.mp4'.format(index)
                export_video(frames, filename)

                if args.export_all:
                    increment_solution(1)
                    if index == 0: # Must have found all solutions
                        running = False
                
            pygame.display.flip()

            if not args.export_all:
                clock.tick(60)
            t += 1
    finally:
        pygame.quit()

