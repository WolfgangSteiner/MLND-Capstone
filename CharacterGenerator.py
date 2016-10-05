#! /usr/bin/python2
from __future__ import print_function
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform
import random
import glob, os, os.path
import numpy as np
import re
from Common import to_categorical
import keras.utils.np_utils to_categorical


font_blacklist = (
    # Linux fonts
    "FiraMono", "Corben", "D050000L","jsMath", "Redacted",
    "RedactedScript", "AdobeBlank", "EncodeSans", "cwTeX", "Droid", "Yinmar", "Lao",
        # Apple fonts
    "Apple Braille", "NISC18030", "Wingdings", "Webdings", "LastResort",
    "Bodoni Ornaments", "Hoefler Text Ornaments", "ZapfDingbats", "Kokonor",
    "Farisi", "Symbol", "Diwan Thuluth", "Diwan")

char_height = 32
char_width = 32
canvas_width = 2 * char_width
canvas_height = 2 * char_height
num_char_columns = 64
num_char_rows = 32
debug = False

def calc_text_size(text, font_tuple):
    font_name, font = font_tuple
    try:
        return font.getsize(text)
    except IOError:
        print("font.getsize failed for font:%s" % font_name)
        raise IOError

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
                font_size = 16
                text_height = 0
                font = None

                while text_height < char_height * 0.9:
                    font = ImageFont.truetype(font=font_file, size=font_size)
                    _,text_height = calc_text_size("0123456789", (font_file, font))
                    font_size += 1

                font_array.append((font_file, font))
                print("adding font: %s, size %d" % (font_file, font_size - 1))
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
num_fonts = len(font_array)

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

random_char.char_array = list("0123456789")

def perspective_transform(char_image):
    (w,h) = char_image.size
    bounding_box = np.array([0,0,0,h,w,h,w,0]) * 2 + (2 * np.random.rand(8) - 1) * w / 4
    transformation = ImageTransform.QuadTransform(bounding_box)
    return char_image.transform((w * 2, h * 2), transformation, resample=Image.BICUBIC)


def rotate(char_image):
    angle = random.randrange(-10,10)
    return char_image.rotate(angle, resample=Image.BICUBIC, expand = 0)


def blur(char_image):
    return char_image.filter(ImageFilter.GaussianBlur(radius=1.5 * random.random()))


def crop(char_image):
    (w,h) = char_image.size
    return char_image.crop((w/4, h/4, w * 3 / 4, h * 3 / 4))


def create_char(font_tuple, char):
    font = font_tuple[1]
    font_name = font_tuple[0]
    background_color, text_color = random_colors()
    text = char

    char_image = create_char_background(background_color)
    draw = ImageDraw.Draw(char_image)

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

    #char_image = perspective_transform(char_image)
    char_image = rotate(char_image)
    char_image = blur(char_image)
    return crop(char_image)


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

        yield np.array(x),to_categorical(y,len(random_char.char_array))


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
