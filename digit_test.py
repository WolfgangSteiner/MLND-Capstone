#! /usr/bin/python2

from __future__ import print_function
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform
import random
import glob, os, os.path
import numpy as np

font_blacklist = ("FiraMono", "Corben", "D050000L","jsMath", "Redacted",
  "RedactedScript", "AdobeBlank", "EncodeSans", "cwTeX", "Droid", "Yinmar",
  "Lao")

def is_font_blacklisted(font_file):
    font_family = os.path.basename(font_file).split(".")[0].split("-")[0]
    return font_family.startswith(font_blacklist)

#load all ttf fonts
font_array = []
for font_file in glob.iglob("/usr/share/fonts/*/*.ttf"):
    if not is_font_blacklisted(font_file):
        font_array.append((font_file, ImageFont.truetype(font=font_file, size=16)))
    else:
        print("Skipping blacklisted font: %s" % font_file)

digit_size = 32
num_digit_columns = 32
num_digit_rows = 32
debug = False

overview_image = Image.new("L", (digit_size * num_digit_columns, digit_size * num_digit_rows), 255)
overview_draw = ImageDraw.Draw(overview_image)

def add_outline(draw, x, y, font, digit, text_color):
    while True:
        outline_color = random.randint(0,255)
        # find an outline color that has a minimum amount of contrast against text_color:
        if abs(text_color - outline_color) > 32:
            break

    draw.text((x-1,y-1), str(digit), font=font, fill=outline_color)
    draw.text((x+1,y-1), str(digit), font=font, fill=outline_color)
    draw.text((x-1,y+1), str(digit), font=font, fill=outline_color)
    draw.text((x+1,y+1), str(digit), font=font, fill=outline_color)

def displacement():
    return np.array([random.random(), random.random()]) * digit_size / 4

def create_digit(font_tuple, digit):
    font = font_tuple[1]
    font_name = font_tuple[0]
    background_color = random.randint(0,255)
    while True:
        text_color = random.randint(0,255)
        # find a text color that has a minimum amount of contrast against background_color:
        if abs(text_color - background_color) > 16:
            break

    digit_image = Image.new("L", (2 * digit_size, 2 * digit_size), background_color)
    draw = ImageDraw.Draw(digit_image)
    try:
        (w,h) = font.getsize(str(digit))
    except IOError:
        print("font.getsize failed for font:%s" % font_name)

    x = 0.5 * (2 * digit_size - w)
    y = 0.5 * (2 * digit_size - h)

    if random.random() > 0.5:
        add_outline(draw, x, y, font, digit, text_color)

    draw.text((x,y), str(digit), font=font, fill=text_color)

    bounding_box = np.array([0,0,0,1,1,1,1,0]) * digit_size * 2 + (2 * np.random.rand(8) - 1) * digit_size / 4
    transformation = ImageTransform.QuadTransform(bounding_box)
    digit_image = digit_image.transform((digit_size * 2, digit_size * 2), transformation, resample=Image.BICUBIC)

    angle = random.randrange(-15,15)
    digit_image = digit_image.rotate(angle, resample=Image.BICUBIC, expand = 0)

    digit_image = digit_image.crop((digit_size/2, digit_size/2, digit_size * 3 / 2, digit_size * 3 / 2))

    digit_image = digit_image.filter(ImageFilter.GaussianBlur(radius=1.5 * random.random()))

    if debug:
        draw = ImageDraw.Draw(digit_image)
        draw.text((0,0), str(digit), font=ImageFont.load_default(), fill=text_color)

    return digit_image

default_font = ImageFont.load_default()

for i in range(0,num_digit_columns):
    for j in range(0,num_digit_rows):
        font_tuple = random.choice(font_array)
        digit_image = create_digit(font_tuple, random.randint(0,9))
        overview_image.paste(digit_image, (digit_size*i, digit_size*j))

        if debug:
            print("%02d/%02d: %s" % (i,j, font_tuple[0]))
            overview_draw.text((i * digit_size, j * digit_size + 22), "%02d/%02d" % (i,j))

overview_image.save("overview.png")
##os.system("eog overview.png")
