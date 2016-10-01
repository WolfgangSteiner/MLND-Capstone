#! /usr/bin/python2
from __future__ import print_function
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform
import random
import glob, os, os.path
import numpy as np
from keras.utils import np_utils
import re


font_blacklist = (
    # Linux fonts
    "FiraMono", "Corben", "D050000L","jsMath", "Redacted",
    "RedactedScript", "AdobeBlank", "EncodeSans", "cwTeX", "Droid", "Yinmar", "Lao",
        # Apple fonts
    "Apple Braille", "NISC18030", "Wingdings", "Webdings", "LastResort",
    "Bodoni Ornaments", "Hoefler Text Ornaments", "ZapfDingbats", "Kokonor",
    "Farisi", "Symbol", "Diwan Thuluth", "Diwan")


# Blacklist symbol fonts and fonts not working with PIL
def is_font_blacklisted(font_file):
    pattern = re.compile("^[A-Z]")
    font_family = os.path.basename(font_file).split(".")[0].split("-")[0]
    return font_family.startswith(font_blacklist) or not pattern.match(font_family)


def is_latin_font(font_subdir):
    try:
        fo = open(font_subdir + "/METADATA.pb", "r")
        return 'subsets: "latin"\n' in fo.readlines()
    except:
        return False

def load_fonts_in_subdir(directory_path, font_array):
    for font_file in glob.iglob(directory_path + "/*.ttf"):
        if not is_font_blacklisted(font_file):
            try:
                font_array.append((font_file, ImageFont.truetype(font=font_file, size=16)))
                print("adding font: %s" % font_file)
            except IOError:
                print("Error loading font: %s" % font_file)


# Collect all ttf fonts in one font location, except those blacklisted.
def find_fonts_in_directory(directory_path):
    font_array = []
    for font_subdir in glob.iglob(directory_path + "/*"):
        if is_latin_font(font_subdir):
            load_fonts_in_subdir(font_subdir, font_array)
        else:
            print("Skipping non-latin fonts in : %s" % font_subdir)

    return font_array


def find_fonts():
    font_array = []
    for font_dir in ("fonts-master/ofl",):
        font_array += find_fonts_in_directory(font_dir)

    return font_array

font_array = find_fonts()

char_height = 24
char_width = 12
canvas_width = 2 * char_width
canvas_height = 2 * char_height
num_char_columns = 64
num_char_rows = 32
debug = False
num_fonts = len(font_array)
current_font = 0

def add_outline(draw, x, y, font, char, text_color):
    while True:
        outline_color = random.randint(0,255)
        # find an outline color that has a minimum amount of contrast against text_color:
        if abs(text_color - outline_color) > 32:
            break

    draw.text((x-1,y-1), char, font=font, fill=outline_color)
    draw.text((x+1,y-1), char, font=font, fill=outline_color)
    draw.text((x-1,y+1), char, font=font, fill=outline_color)
    draw.text((x+1,y+1), char, font=font, fill=outline_color)

def displacement():
    return np.array([random.random() * char_width / 4, random.random() * char_height / 4])

def random_colors():
    background_color = random.randint(0,255)
    while True:
        text_color = random.randint(0,255)
        # find a text color that has a minimum amount of contrast against background_color:
        if abs(text_color - background_color) > 32:
            return background_color, text_color

def create_char_background(background_color):
    noise = (np.random.rand(canvas_height, canvas_width) - 0.5) * random.randint(0,32) + background_color
    return Image.fromarray(noise).convert("L")

def random_char():
    return random.choice(random_char.char_array)

random_char.char_array = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def calc_text_size(text, font_tuple):
    font_name, font = font_tuple
    try:
        return font.getsize(text)
    except IOError:
        print("font.getsize failed for font:%s" % font_name)
        raise IOError


def create_char(font_tuple, char):
    font = font_tuple[1]
    font_name = font_tuple[0]

    background_color, text_color = random_colors()
    char_image = create_char_background(background_color)
    draw = ImageDraw.Draw(char_image)

    text = char

    (w,h) = calc_text_size(text, font_tuple)
    x = 0.5 * (canvas_width - w)
    y = 0.5 * (canvas_height - h)

    if random.random() > 0.25:
        text = random_char() + text
        (w2,h2) = calc_text_size(text, font_tuple)
        x -= (w2 - w)

    if random.random() > 0.25:
        text = text + random_char()

    if random.random() > 0.5:
        add_outline(draw, x, y, font, text, text_color)

    draw.text((x,y), text, font=font, fill=text_color)

    bounding_box = np.array([0,0,0,char_height,char_width,char_height,char_width,0]) * 2 + (2 * np.random.rand(8) - 1) * char_width / 4
    transformation = ImageTransform.QuadTransform(bounding_box)
    char_image = char_image.transform((char_width * 2, char_height * 2), transformation, resample=Image.BICUBIC)

    angle = random.randrange(-15,15)
    char_image = char_image.rotate(angle, resample=Image.BICUBIC, expand = 0)
    char_image = char_image.filter(ImageFilter.GaussianBlur(radius=1.5 * random.random()))
    char_image = char_image.crop((char_width/2, char_height/2, char_width * 3 / 2, char_height * 3 / 2))
    return char_image

def create_random_char():
    font_tuple = random.choice(font_array)
    return create_char(font_tuple, random_char())

def CharacterGenerator(batchsize):
    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            font_tuple = random.choice(font_array)
            char = random_char()
            char_image = create_char(font_tuple, char)
            x.append(np.array(char_image).reshape(char_height,char_width,1).astype('float32') / 255.0)
            y.append(random_char.char_array.index(char))

        yield np.array(x),np_utils.to_categorical(y,nb_classes=len(random_char.char_array))


if __name__ == "__main__":
    overview_image = Image.new("L", (char_width * num_char_columns, char_height * num_char_rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)
    generator = CharacterGenerator(num_char_columns)

    for j in range(0,num_char_rows):
        batch = generator.next()
        for i in range(0,num_char_columns):
            font_tuple = random.choice(font_array)
            char = random_char()
            overview_image.paste(create_char(font_tuple, char), (char_width*i, char_height*j))

            #overview_image.paste(Image.fromarray((batch[0][i].reshape(char_height,char_width) * 255).astype('uint8'),mode="L"), (char_width*i, char_height*j))

            if debug:
                print("%02d/%02d: %s" % (j,i, font_tuple[0]))
                overview_draw.text((i * char_width, j * char_height + 10), char)
                overview_draw.text((i * char_width, j * char_height + 38), "%02d/%02d" % (j,i))

    overview_image.save("overview.png")
