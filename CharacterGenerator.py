#! /usr/bin/python2
from __future__ import print_function
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform
import random
import glob, os, os.path
import numpy as np
from sys import platform
from keras.utils import np_utils

# Get list of font locations. Additional fonts can be added to "fonts" in the
# working directory.
def font_dirs():
    if platform == "linux2":
        return ("/usr/share/fonts/*", "./fonts")
    elif platform == "darwin":
        return ("/Library/Fonts", "/System/Library/Fonts", os.path.expanduser("~/Library/Fonts"), "./fonts")
    else:
        raise RuntimeError('unsupported OS')

# Blacklist symbol fonts and fonts not working with PIL
def is_font_blacklisted(font_file):
    font_blacklist = (
        # Linux fonts
        "FiraMono", "Corben", "D050000L","jsMath", "Redacted",
        "RedactedScript", "AdobeBlank", "EncodeSans", "cwTeX", "Droid", "Yinmar", "Lao",
        # Apple fonts
        "Apple Braille", "NISC18030", "Wingdings", "Webdings", "LastResort",
        "Bodoni Ornaments", "Hoefler Text Ornaments", "ZapfDingbats", "Kokonor")

    font_family = os.path.basename(font_file).split(".")[0].split("-")[0]
    return font_family.startswith(font_blacklist)

# Collect all ttf fonts in one font location, except those blacklisted.
def find_fonts_in_directory(directory_path):
    font_array = []
    for font_file in glob.iglob(directory_path + "/*.ttf"):
        if not is_font_blacklisted(font_file):
            try:
                font_array.append((font_file, ImageFont.truetype(font=font_file, size=16)))
                print("adding font: %s" % font_file)
            except IOError:
                print("Error loading font: %s" % font_file)
        else:
            print("Skipping blacklisted font: %s" % font_file)

    return font_array

def find_fonts():
    font_array = []
    for font_dir in font_dirs():
        font_array += find_fonts_in_directory(font_dir)

    return font_array

font_array = find_fonts()

digit_size = 24
num_digit_columns = 32
num_digit_rows = 32
debug = False
num_fonts = len(font_array)
current_font = 0

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


def random_colors():
    background_color = random.randint(0,255)
    while True:
        text_color = random.randint(0,255)
        # find a text color that has a minimum amount of contrast against background_color:
        if abs(text_color - background_color) > 32:
            return background_color, text_color
    
def create_digit_background(background_color):
    noise = (np.random.rand(2 * digit_size, 2 * digit_size) - 0.5) * random.randint(0,32) + background_color
    return Image.fromarray(noise).convert("L")

        
def create_digit(font_tuple, digit):
    font = font_tuple[1]
    font_name = font_tuple[0]

    background_color, text_color = random_colors()
    digit_image = create_digit_background(background_color)
    draw = ImageDraw.Draw(digit_image)

    try:
        (w,h) = font.getsize(str(digit))
    except IOError:
        print("font.getsize failed for font:%s" % font_name)
        raise IOError

    x = 0.5 * (2 * digit_size - w)
    y = 0.5 * (2 * digit_size - h)

    if random.random() > 1.0:
        add_outline(draw, x, y, font, digit, text_color)

    draw.text((x,y), str(digit), font=font, fill=text_color)

    bounding_box = np.array([0,0,0,1,1,1,1,0]) * digit_size * 2 + (2 * np.random.rand(8) - 1) * digit_size / 4
    transformation = ImageTransform.QuadTransform(bounding_box)
    digit_image = digit_image.transform((digit_size * 2, digit_size * 2), transformation, resample=Image.BICUBIC)

    angle = random.randrange(-15,15)
    digit_image = digit_image.rotate(angle, resample=Image.BICUBIC, expand = 0)

    digit_image = digit_image.filter(ImageFilter.GaussianBlur(radius=1.5 * random.random()))

    digit_image = digit_image.crop((digit_size/2, digit_size/2, digit_size * 3 / 2, digit_size * 3 / 2))

    if False:
        draw = ImageDraw.Draw(digit_image)
        draw.text((0,0), str(digit), font=ImageFont.load_default(), fill=text_color)

    return digit_image


def create_random_digit():
    font_tuple = random.choice(font_array)
    return create_digit(font_tuple, random.randint(0,9))


def CharacterGenerator(batchsize):
    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            font_tuple = random.choice(font_array)
            digit = random.randint(0,9)
            digit_image = create_digit(font_tuple, digit)
            x.append(np.array(digit_image).reshape(24,24,1).astype('float32') / 255.0)
            y.append(digit)

        yield np.array(x),np_utils.to_categorical(y,nb_classes=10)


if __name__ == "__main__":
    overview_image = Image.new("L", (digit_size * num_digit_columns, digit_size * num_digit_rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)

    generator = CharacterGenerator(32)
    for i in range(0,num_digit_columns):
        j = 0
        batch = generator.next()
        for j in range(0,32):
            overview_image.paste(Image.fromarray((batch[0][j].reshape(24,24) * 255).astype('uint8'),mode="L"), (digit_size*i, digit_size*j))
#
            if debug:
                print("%02d/%02d: %s" % (i,j, font_tuple[0]))
                overview_draw.text((i * digit_size, j * digit_size + 22), "%02d/%02d" % (i,j))

            
    overview_image.save("overview.png")
