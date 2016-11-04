#! /usr/bin/python2
from __future__ import print_function
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform, ImageChops
import random
import glob, os, os.path
import pickle
import numpy as np
import re
import math
import cairo
from keras.utils.np_utils import to_categorical

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
num_char_columns = 16
num_char_rows = 32
debug = False

def calc_text_size(text, font_tuple):
    font_name, font = font_tuple
    try:
        (w,h) = font.getsize(text)
        h -= font.getoffset(text)[1]
        return (w,h)
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

                max_font_size = font_size - 1

                font_array.append((font_file, max_font_size))
                print("adding font: %s, size %d" % (font_file, max_font_size))
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
    for font_dir in ("fonts-master/ofl", "fonts-master/apache", ):
        font_array += find_fonts_in_directory(font_dir)

    return font_array

try:
    print("Loading fonts from font_cache.pickle ...")
    file = open('font_cache.pickle', 'rb')
    font_array = pickle.load(file)
except IOError:
    font_array = find_fonts()
    file = open('font_cache.pickle', 'wb')
    print ("Writing font_cache.pickle ...")
    pickle.dump(font_array, file, -1)

def random_font(options={}):
    min_size = options.get('min_size', 0.75)
    max_size = options.get('max_size', 1.0)
    font_name, max_font_size = random.choice(font_array)
    size = random.randint(int(max_font_size * min_size), int(max_font_size * max_size))
    return (font_name, ImageFont.truetype(font_name, size))

def get_color(color, alpha=255):
    return (color,color,color,alpha)

def add_outline(draw, x, y, font, char, text_color):
    while True:
        outline_color = random.randint(0,255)
        # find an outline color that has a minimum amount of contrast against text_color:
        if abs(text_color - outline_color) > 32:
            break

    draw_text(draw, x-1, y-1, char, font, outline_color)
    draw_text(draw, x+1, y-1, char, font, outline_color)
    draw_text(draw, x-1, y+1, char, font, outline_color)
    draw_text(draw, x+1, y+1, char, font, outline_color)


def add_shadow(image, x, y, font, char, text_color):
    shadow_image = Image.new('RGBA', image.size, (0,0,0,0))
    shadow_layer = Image.new('RGBA', image.size, (0,0,0,255))
    shadow_image = Image.composite(shadow_layer, shadow_image, image.split()[-1])
    result = Image.new('RGBA', image.size, (0,0,0,0))
    result.paste(shadow_image, (random.randint(-3,3), random.randint(-3,3)))

    for n in (0, 10):
        result = result.filter(ImageFilter.BLUR)

    return result

def displacement():
    return np.array([random.random() * char_width / 4, random.random() * char_height / 4])

def random_background_color(text_color, min_color_delta=32):
    while True:
        background_color = random.randint(0,255)
        # find a text color that has a minimum amount of contrast against background_color:
        if abs(text_color - background_color) > min_color_delta:
            return background_color


def draw_line(draw, p1, p2, color, width, alpha=255):
    draw.line((p1[0],p1[1],p2[0],p2[1]), fill=get_color(color, alpha=alpha), width=width)

def draw_text(draw, x, y, text, font, color):
    draw.text((x,y), text, font=font, fill=get_color(color))

def draw_random_line(draw, text_color, min_color_delta, oversampling=4):
    p1 = np.random.random(2) * char_width * oversampling
    angle = random.random() * math.pi
    length = random.random() * char_width * oversampling
    width = random.randint(1, char_width * oversampling)
    color = random_background_color(text_color, min_color_delta=min_color_delta)
    alpha = random.randint(64,255)
    p2 = p1 + np.array([math.cos(angle), math.sin(angle)]) * length
    draw_line(draw, p1, p2, color, width, alpha=255)

def add_random_lines(draw, text_color, min_color_delta, oversampling=4):
    while True:
        draw_random_line(draw, text_color, min_color_delta, oversampling=oversampling)
        if random.random() > 0.96:
            break

def add_noise(image, options={}):
    min_noise = options.get('min_noise', 8)
    max_noise = options.get('max_noise', 8)
    w,h = image.size
    noise = (np.random.rand(w,h) - 0.5) * (min_noise + random.randint(0,max_noise - min_noise))
    im_array = np.array(image.convert('L')).astype(np.float32)
    im_array = np.clip(im_array + noise, 0.0, 255.0)
    return Image.fromarray(im_array).convert('L')

def create_char_background(text_color, background_color, min_color_delta):
    oversampling = 2
    image = Image.new('RGBA', (canvas_width * oversampling, canvas_height * oversampling), get_color(background_color))
    draw = ImageDraw.Draw(image, 'RGBA')
    add_random_lines(draw, text_color, min_color_delta, oversampling)
    image = image.resize((canvas_width, canvas_height), resample=Image.LANCZOS)
    image = blur(image, {'min_blur':0.125, 'max_blur':0.5})
    return image

def random_char():
    return random.choice(random_char.char_array)

random_char.char_array = list("0123456789")

def perspective_transform(char_image):
    (w,h) = char_image.size
    bounding_box = np.array([0,0,0,h,w,h,w,0]) + (np.random.rand(8) - 0.5) * w / 4
    transformation = ImageTransform.QuadTransform(bounding_box)
    return char_image.transform((w, h), transformation, resample=Image.BICUBIC)

def rotate(char_image, options={}):
    max_rotation=options.get('max_rotation', 5)
    angle = random.randrange(-max_rotation,max_rotation)
    return char_image.rotate(angle, resample=Image.BICUBIC, expand = 0)

def blur(char_image, options={}):
    min_blur = options.get("min_blur", 1.0)
    max_blur = options.get('max_blur', 2.0)
    return char_image.filter(ImageFilter.GaussianBlur(radius=(min_blur + (max_blur - min_blur) * random.random())))

def crop(char_image):
    (w,h) = char_image.size
    return char_image.crop((w/4, h/4, w * 3 / 4, h * 3 / 4))

def normalize(char_image, factor):
    array = np.array(char_image).astype(np.float32)
    m = np.mean(array)
    s = np.std(array)
    array = np.clip(array, 0.0, 255.0)
    return Image.fromarray(array).convert('L')

def create_char(font_tuple, char, options={}):
    font = font_tuple[1]
    font_name = font_tuple[0]
    min_color_delta = options.get('min_color_delta', 32)
    text_color = random.randint(0,255)
    background_color = random_background_color(text_color, min_color_delta=min_color_delta)
    text = char

    image = create_char_background(text_color, background_color, min_color_delta)
    char_image = Image.new('RGBA', (canvas_width, canvas_height), (0,0,0,0))

    (w,h) = calc_text_size(text, font_tuple)
    x = 0.5 * (canvas_width - w)
    y = 0.5 * (canvas_height - h)

    if random.random() > 0.5:
        text = random_char() + text
        (w2,h2) = calc_text_size(text, font_tuple)
        x -= (w2 - w)

    if random.random() > 0.5:
        text = text + random_char()

    x += (random.random() - 0.5) * 0.5 * (char_width - w)
    y += (random.random() - 0.5) * (char_height - h)
    y -= font.getoffset(text)[1]

    draw = ImageDraw.Draw(char_image)

    if random.random() > 0.5:
        add_outline(draw, x, y, font, text, text_color)

    draw_text(draw, x, y, text, font, text_color)

    if random.random() > 0.5:
        shadow_image = add_shadow(char_image, x, y, font, text, text_color)
        image = Image.alpha_composite(image, shadow_image)

    char_image = Image.alpha_composite(image, char_image)
    char_image = rotate(char_image, options)
    char_image = perspective_transform(char_image)
    char_image = crop(char_image)
    char_image = blur(char_image, options)
    char_image = add_noise(char_image, options)
    return char_image


def create_random_char(options={}):
    font_tuple = random_font(options)
    return create_char(font_tuple, random_char())


def CharacterGenerator(batchsize, options={}):
    mean = options.get('mean', None)
    std = options.get('std', None)

    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            font_tuple = random_font(options)
            char = random_char()
            char_image = create_char(font_tuple, char, options)
            char_data = np.array(char_image).astype('float32')

            if mean == None:
                mean = np.mean(char_data, axis=(0,1))

            if std == None:
                std = np.std(char_data, axis=(0,1))

            char_data = (char_data - mean) / std

            x.append(char_data.reshape(char_height,char_width,1))
            y.append(random_char.char_array.index(char))

        yield np.array(x),to_categorical(y,len(random_char.char_array))


if __name__ == "__main__":
    overview_image = Image.new("L", (char_width * num_char_columns, char_height * num_char_rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)
    options={'min_color_delta':16.0, 'min_blur':0.5, 'max_blur':2.0, 'max_rotation':5.0, 'min_noise':4, 'max_noise':4}
    for j in range(0,num_char_rows):
        for i in range(0,num_char_columns):
            font_tuple=random_font(options)
            char = random_char()
            overview_image.paste(create_char(font_tuple, char, options), (char_width*i, char_height*j))

            #overview_image.paste(Image.fromarray((batch[0][i].reshape(char_height,char_width) * 255).astype('uint8'),mode="L"), (char_width*i, char_height*j))

            if debug:
                print("%02d/%02d: %s" % (j,i, font_tuple[0]))
                overview_draw.text((i * char_width, j * char_height + 10), char)
                overview_draw.text((i * char_width, j * char_height + 38), "%02d/%02d" % (j,i))

    overview_image.save("overview.png")
