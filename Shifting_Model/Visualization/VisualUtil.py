import math
from PIL import Image, ImageDraw, ImageFont
import configparser
import pandas as pd
import numpy as np
import cairo

# Color Constants as RGBA tuples:
NAVY_DARK    = (0.05, 0.14, 0.25, 1.00) # Auburn Navy Dark
NAVY_LIGHT   = (0.81, 0.83, 0.85, 1.00) # Auburn Navy Light
ORANGE_DARK  = (0.91, 0.38, 0.00, 1.00) # Auburn Orange Dark
ORANGE_LIGHT = (0.98, 0.87, 0.80, 1.00) # Auburn Orange Light
WHITE        = (1.00, 1.00, 1.00, 1.00) # White
BLACK        = (0.00, 0.00, 0.00, 1.00) # Black
TRANSPARENT  = (0.00, 0.00, 0.00, 0.00) # Transparent

# Global Variable Declarations
DISTANCE_TO_PLATE_  =  60.5
DISTANCE_TO_GRASS_  =  95.0
DISTANCE_TO_FENCE_  = 339.5
image_scale_factor  =   2
cushion             =  10   # Extra space around the field to allow for thicker lines / outlines. Uniform on all sides.
outline_thickness   =  12
line_thickness      =   6

config = configparser.ConfigParser()
config.read('Data//config.ini')

# Creates an image of a baseball field with slices colored based on the odds of being hit into. Also displays the percent chance on each slice.
def visualizeData(infieldStats, outfieldStats, filename):
    initializeFieldVariables()
    infield_slices  = infieldStats.__len__()
    outfield_slices = outfieldStats.__len__()
    #print(config['VISUAL']['Heatmap']=='True')
    # Tweakable params for the heatmap
    heatmap_horz_density = int(config['VISUAL']['HorizontalDensity'])
    heatmap_vert_density = int(config['VISUAL']['VerticalDensity'])

    # Create the field image
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, FIELD_WIDTH, FIELD_HEIGHT)
    draw = cairo.Context(surface)

    # Draw the slices & layer field lines on top
    if(config['VISUAL']['RenderOutfield']=='True'):
        if(config['VISUAL']['Heatmap']=='True'):
            fillHeatmap(draw, convertToMoundSpace(outfieldStats, heatmap_vert_density, heatmap_horz_density), NAVY_LIGHT, NAVY_DARK)
            fillSlices(draw, infield_slices,  infieldStats,  DISTANCE_TO_GRASS, INFIELD_ARC,  ORANGE_LIGHT, ORANGE_DARK)
            #debugPlot(draw, outfieldStats, WHITE)
            drawHeatmapLattice(draw, heatmap_vert_density, heatmap_horz_density, outline_thickness/2, WHITE)
            drawField(draw, infield_slices, 0, outline_thickness, WHITE)
            drawHeatmapLattice(draw, heatmap_vert_density, heatmap_horz_density, line_thickness/2, BLACK)
            drawField(draw, infield_slices, 0, line_thickness, BLACK)
        else:
            fillSlices(draw, outfield_slices, outfieldStats, DISTANCE_TO_FENCE, OUTFIELD_ARC, NAVY_LIGHT,   NAVY_DARK)
            fillSlices(draw, infield_slices,  infieldStats,  DISTANCE_TO_GRASS, INFIELD_ARC,  ORANGE_LIGHT, ORANGE_DARK)
            drawField(draw, infield_slices, outfield_slices, outline_thickness, WHITE)
            drawField(draw, infield_slices, outfield_slices, line_thickness, BLACK)
    else:
        fillSlices(draw, infield_slices,  infieldStats,  DISTANCE_TO_FENCE, OUTFIELD_ARC,  ORANGE_LIGHT, ORANGE_DARK)
        drawOnlyInfield(draw, infield_slices)
    #print('Output/' + filename + '.png')
    surface.write_to_png('Output/' + filename + '.png')

    # Write text on top of the image
    image = Image.open('Output/' + filename + '.png')
    if(config['VISUAL']['RenderOutfield']=='True'):
        image = addPercents(image, infield_slices,  infieldStats,  DISTANCE_TO_GRASS)
        if(config['VISUAL']['Heatmap']=='False'):
            image = addPercents(image, outfield_slices, outfieldStats, DISTANCE_TO_FENCE)
    else:
        image = addPercents(image, infield_slices,  infieldStats,  DISTANCE_TO_FENCE)

    image.save('Output/' + filename + '.png')


# Top left corner is (0,0)
# Bottom right corner is (field_width, field_height)
def initializeFieldVariables():
    global DISTANCE_TO_PLATE, DISTANCE_TO_GRASS, DISTANCE_TO_FENCE, PLATE, MOUND, FOULL, FOULR, OUTFIELD_ARC, INFIELD_ARC, FIELD_HEIGHT, FIELD_WIDTH
    DISTANCE_TO_PLATE = DISTANCE_TO_PLATE_ * image_scale_factor
    DISTANCE_TO_GRASS = DISTANCE_TO_GRASS_ * image_scale_factor
    DISTANCE_TO_FENCE = DISTANCE_TO_FENCE_ * image_scale_factor

    foul_line_intercept = getIntersection((0, DISTANCE_TO_PLATE, DISTANCE_TO_FENCE), (0,0), math.radians(45))
    base_line_intercept = getIntersection((0, DISTANCE_TO_PLATE, DISTANCE_TO_GRASS), (0,0), math.radians(45))
    FIELD_HEIGHT = int(math.ceil((DISTANCE_TO_PLATE + DISTANCE_TO_FENCE)) + (2*cushion))
    FIELD_WIDTH  = int(math.ceil(foul_line_intercept[0] * 2) + (2*cushion))
    base_width   = int(math.ceil(base_line_intercept[0] * 2))
    #print('Field Height: ' + str(field_height) + '\nField Width: ' + str(field_width))

    PLATE = (FIELD_WIDTH/2, FIELD_HEIGHT - cushion)
    MOUND = (FIELD_WIDTH/2, FIELD_HEIGHT - cushion - DISTANCE_TO_PLATE)
    FOULL = (cushion, FIELD_HEIGHT - cushion - math.ceil(foul_line_intercept[1]))
    FOULR = (FIELD_WIDTH - cushion, FIELD_HEIGHT - cushion - math.ceil(foul_line_intercept[1]))
    OUTFIELD_ARC = math.asin((FIELD_WIDTH - 2*cushion)/(2*DISTANCE_TO_FENCE))
    INFIELD_ARC  = math.asin((base_width)/(2*DISTANCE_TO_GRASS))

def drawField(draw, infield, outfield, thick, color):
    drawFieldLines(draw, thick, color)
    drawInfieldSliceLines (draw, thick, color, infield)
    drawOutfieldSliceLines(draw, thick, color, outfield)
    drawFieldSplit(draw, DISTANCE_TO_GRASS, thick, color)

def drawOnlyInfield(draw, infield):
    drawFieldLines(draw, outline_thickness, WHITE)
    drawInfieldSliceLines (draw, outline_thickness, WHITE, infield)
    drawOutfieldSliceLines(draw, outline_thickness, WHITE, infield)
    drawFieldLines(draw,  line_thickness, BLACK)
    drawInfieldSliceLines (draw,  line_thickness, BLACK, infield)
    drawOutfieldSliceLines(draw,  line_thickness, BLACK, infield)

def drawFieldLines(draw, thick, color):
    draw.move_to(PLATE[0], PLATE[1])
    draw.arc(MOUND[0], MOUND[1], DISTANCE_TO_FENCE, -math.pi/2 - OUTFIELD_ARC, -math.pi/2 + OUTFIELD_ARC)
    draw.close_path()
    draw.set_line_width(thick)
    draw.set_source_rgba(color[0], color[1], color[2], color[3])
    draw.stroke()

def drawFieldSplit(draw, distance, thick, color):
    draw.arc(MOUND[0], MOUND[1], distance, -math.pi/2 - getArc(distance), -math.pi/2 + getArc(distance))
    draw.set_line_width(thick)
    draw.set_source_rgba(color[0], color[1], color[2], color[3])
    draw.stroke()

def drawInfieldSliceLines(draw, thick, color, slices):
    if(slices == 0):
        return
    for i in range(1, slices):
        start = (PLATE[0], PLATE[1])
        end   = flip_y(getIntersection(flip_y((MOUND[0], MOUND[1], DISTANCE_TO_GRASS)), flip_y((PLATE[0], PLATE[1])), math.radians(45 + (i * 90 / slices))))
        drawSliceLine(draw, start, end, thick, color)

def drawOutfieldSliceLines(draw, thick, color, slices):
    if(slices == 0):
        return
    for i in range(1, slices):
        start  = flip_y(getIntersection(flip_y((MOUND[0], MOUND[1], DISTANCE_TO_GRASS)), flip_y((PLATE[0], PLATE[1])), math.radians(45 + (i * 90 / slices))))
        end    = flip_y(getIntersection(flip_y((MOUND[0], MOUND[1], DISTANCE_TO_FENCE)), flip_y((PLATE[0], PLATE[1])), math.radians(45 + (i * 90 / slices))))
        drawSliceLine(draw, start, end, thick, color)

def drawSliceLine(draw, start, end, thick, color):
    draw.move_to(start[0], start[1])
    draw.line_to(end[0], end[1])
    draw.set_line_width(thick)
    draw.set_source_rgba(color[0], color[1], color[2], color[3])
    draw.stroke()

def fillSlices(draw, slices, percentages, arc_distance, arc_angle, color1, color2):
    if(slices == 0):
        return
    maxOdds = max(percentages)
    angle_to = -math.pi/2 - arc_angle
    angle_diff = 2 * arc_angle / slices
    for i in range(0, slices):
        angle_from = angle_to
        angle_to += angle_diff
        sliceColor = blendColors(color1, color2, percentages[i]/maxOdds)
        drawFilledSlice(draw, angle_from, angle_to, sliceColor, arc_distance)
    
def drawFilledSlice(draw, angle_from, angle_to, color, radius):
    draw.move_to(PLATE[0], PLATE[1])
    draw.arc(MOUND[0], MOUND[1], radius, angle_from, angle_to)
    draw.close_path()
    draw.set_source_rgba(color[0], color[1], color[2], color[3])
    draw.fill()

def drawText(image, text, position):
    useFont = ImageFont.truetype("Visualization/Fonts/SweetSansProRegular.otf", 30)
    draw = ImageDraw.Draw(image)
    draw.text((position[0], position[1]), text, font=useFont, fill=color10to255(BLACK), align="center", anchor="mm", stroke_width=3, stroke_fill=color10to255(WHITE))

def drawHeatmapLattice(draw, vert_density, horz_density, thick, color):
    # Decide density of heatmap (goes from 150 to 400, so test value = 50 so each section is 50 feet)
    drawVertLattice(draw, vert_density, thick, color)
    drawHorzLattice(draw, horz_density, thick, color)
    #drawFilledSlice(draw, -math.pi/2 - OUTFIELD_ARC, -math.pi/2 + OUTFIELD_ARC, color1, DISTANCE_TO_FENCE)
        
def drawVertLattice(draw, density, thick, color):
    start = DISTANCE_TO_GRASS
    end = DISTANCE_TO_FENCE
    diff = (end - start) / density
    start += diff
    while(start < end):
        drawFieldSplit(draw, start, thick, color)
        start += diff

def drawHorzLattice(draw, density, thick, color):
    start = 0
    end = 90
    diff = (end - start) / density
    start += diff
    while(start < end):
        intersectionAtGrass = getIntersection((MOUND[0], MOUND[1], DISTANCE_TO_GRASS), (PLATE[0], PLATE[1]), math.radians(-135 + start))
        intersectionAtFence = getIntersection((MOUND[0], MOUND[1], DISTANCE_TO_FENCE), (PLATE[0], PLATE[1]), math.radians(-135 + start))
        drawSliceLine(draw, intersectionAtGrass, intersectionAtFence, thick, color)
        start += diff

def fillHeatmap(draw, heatmap, color1, color2):
    # Draw slices from back to front, left to right. This will layer properly.
    max = np.max(heatmap)
    for i in range(0, heatmap.__len__()):
        distance   = DISTANCE_TO_FENCE - ((i) * (DISTANCE_TO_FENCE - DISTANCE_TO_GRASS)/heatmap.__len__())
        minAngle = -math.pi/2 - getArc(distance)
        maxAngle = -math.pi/2 + getArc(distance)
        diff = maxAngle - minAngle
        for j in range(0, heatmap[0].__len__()):
            sliceStart = minAngle + ( j    * (diff / heatmap[0].__len__()))
            sliceEnd   = minAngle + ((j+1) * (diff / heatmap[0].__len__()))
            drawFilledSlice(draw, sliceStart, sliceEnd, blendColors(color1, color2, heatmap[heatmap.__len__() - (i+1)][j]/max), distance)

def debugPlot(draw, stats, color):
    for i in range(0, stats.__len__()):
        drawPoint(draw, stats[i][0], stats[i][1], color)

def drawPoint(draw, bearing, distance, color):
    pos = getIntersection((PLATE[0], PLATE[1], distance*image_scale_factor), (PLATE[0], PLATE[1]), math.radians(-90 + bearing))
    #print('Angle: ' + str(math.degrees(bearing)) + '\nDistance: ' + str(distance) + '\nX: ' + str(pos[0]) + '\nY: ' + str(pos[1]) + '\n')
    draw.move_to(pos[0], pos[1])
    draw.arc(pos[0], pos[1], 5, 0, 2*math.pi)
    draw.set_source_rgba(color[0], color[1], color[2], color[3])
    draw.fill()

def addPercents(image, slices, percentages, arc_distance):
    for i in range(0, slices):
        pos = flip_xy(getIntersection(flip_y((MOUND[0], MOUND[1], arc_distance * 0.75)), flip_y((PLATE[0], PLATE[1])), math.radians(45 + ((i+0.5) * 90 / slices))))
        drawText(image, cleanNumber(percentages[i]), pos)
    return image

# Blends two RGBA colors together based on a ratio (0..1)
def blendColors(color1, color2, ratio):
    blend = [(color1[i] * (1 - ratio) + color2[i] * ratio) for i in range(4)]
    return tuple(blend)

def cleanNumber(num):
    clean = int(round(num*100))
    return str(clean) + "%"

def convertToMoundSpace(stats, vert_density, horz_density):
    # Solve for which horizontal slice the point is in (angle)
    # stats[0] = angle from -45 to 45
    # stats[1] = distance from 150 to 400
    heatmap = np.zeros((vert_density, horz_density))
    for i,stat in enumerate(stats):
        vert = math.floor(i/5) #math.floor((stat - 150)/(250 / vert_density))
        horz = i - (vert*5) #math.floor((stat + 45)/(90 / horz_density))
        heatmap[vert][horz] = stat #+= 1
    return heatmap

# Circle = (x, y, r)
def getIntersection(circle, line_start, angle):
    x = line_start[0] - circle[0]
    y = line_start[1] - circle[1]
    r = circle[2]
    dirX = math.cos(angle)
    dirY = math.sin(angle)

    # Split function for debugging
    xydir = x*dirX + y*dirY
    dir2  = dirX**2 + dirY**2
    rxy   = r**2 - x**2 - y**2
    root  = xydir**2 + dir2*rxy

    t = (math.sqrt(root) - xydir) / dir2
    intersection = (x + dirX*t, y + dirY*t)
    intersection = (circle[0] + intersection[0], circle[1] + intersection[1])
    #print('X intersection: ' + str(intersection[0]) + '\nY intersection: ' + str(intersection[1]))
    return intersection

def getArc(radius):
    a = quadratic(2, -2*DISTANCE_TO_PLATE, DISTANCE_TO_PLATE**2 - radius**2)
    a = math.sqrt(2*a**2) # Gets magnitude of a, which before is just the x or y coordinate (x=y, so mag = sqrt(2x^2))
    b = a * math.sqrt(2)
    result = math.asin(b/(2*radius)) # This is half the angle made by the mound triangle, using half so we can center around -pi/2
    return result

def quadratic(a, b, c):
    return (-b + math.sqrt(b**2 - 4*a*c)) / (2*a)

def flip_y(coords):
    if(coords.__len__() == 2):
        return (coords[0], FIELD_HEIGHT - coords[1])
    if(coords.__len__() == 3):
        return (coords[0], FIELD_HEIGHT - coords[1], coords[2])
    
def flip_x(coords):
    if(coords.__len__() == 2):
        return (FIELD_WIDTH - coords[0], coords[1])
    if(coords.__len__() == 3):
        return (FIELD_WIDTH - coords[0], coords[1], coords[2])
    
def flip_xy(coords):
    return flip_x(flip_y(coords))
    
def color10to255(color):
    return (int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255))
