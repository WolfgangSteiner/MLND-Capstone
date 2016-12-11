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
import string
from keras.utils.np_utils import to_categorical
from CharacterSource import NumericCharacterSource, AlphaNumericCharacterSource
from FontSource import FontSource

char_height = 32
char_width = 32
canvas_width = 2 * char_width
canvas_height = 2 * char_height
num_char_columns = 16
num_char_rows = 32
debug = False
char_source = NumericCharacterSource()
font_source = FontSource()


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


def displacement(char_width, char_height):
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
    draw.text((x,y), text, font=font.image_font, fill=get_color(color))


def draw_random_line(canvas_width, canvas_height, draw, text_color, min_color_delta, oversampling=4):
    p1 = np.random.random(2) * canvas_width / 2 * oversampling
    angle = random.random() * math.pi
    length = random.random() * canvas_width / 2 * oversampling
    width = random.randint(1, canvas_width / 2 * oversampling)
    color = random_background_color(text_color, min_color_delta=min_color_delta)
    alpha = random.randint(64,255)
    p2 = p1 + np.array([math.cos(angle), math.sin(angle)]) * length
    draw_line(draw, p1, p2, color, width, alpha=255)


def add_random_lines(canvas_width, canvas_height, draw, text_color, min_color_delta, oversampling=4):
    while random.random() < 0.95:
        draw_random_line(canvas_width, canvas_height, draw, text_color, min_color_delta, oversampling=oversampling)


def add_noise(image, options={}):
    min_noise = options.get('min_noise', 8)
    max_noise = options.get('max_noise', 8)
    w,h = image.size
    noise = (np.random.rand(h,w) - 0.5) * (min_noise + random.randint(0,max_noise - min_noise))
    im_array = np.array(image.convert('L')).astype(np.float32)
    im_array = np.clip(im_array + noise, 0.0, 255.0)
    return Image.fromarray(im_array).convert('L')


def create_char_background(width, height, text_color, background_color, min_color_delta, options={}):
    add_background_lines = options.get('add_background_lines', True)
    oversampling = options.get('oversampling', 2)

    if add_background_lines:
        image = Image.new('RGBA', (width * oversampling, height * oversampling), get_color(background_color))
        draw = ImageDraw.Draw(image, 'RGBA')
        add_random_lines(width, height, draw, text_color, min_color_delta, oversampling)
        image = image.resize((width, height), resample=Image.LANCZOS)
        image = blur(image, {'min_blur':0.125, 'max_blur':0.5})
    else:
        image = Image.new('RGBA', (width, height), get_color(background_color))

    return image


def perspective_transform(char_image):
    (w,h) = char_image.size
    bounding_box = np.array([0,0,0,h,w,h,w,0]) + (np.random.rand(8) - 0.5) * w / 4
    transformation = ImageTransform.QuadTransform(bounding_box)
    return char_image.transform((w, h), transformation, resample=Image.BICUBIC)


def rotate(char_image, options={}):
    max_rotation=options.get('max_rotation', 5.0)

    if max_rotation > 0:
        angle = np.random.normal(0.0, max_rotation)
        return char_image.rotate(angle, resample=Image.BICUBIC, expand = 0)
    else:
        return char_image


def blur(char_image, options={}):
    min_blur = options.get("min_blur", 1.0)
    max_blur = options.get('max_blur', 2.0)
    return char_image.filter(ImageFilter.GaussianBlur(radius=(min_blur + (max_blur - min_blur) * random.random())))


def crop(char_image, width=None, rescale=True):
    (w,h) = char_image.size
    y1 = h / 4
    y2 = h * 3 / 4

    if width != None:
        x1 = w / 2 - width / 2
        x2 = w / 2 + width / 2
    else:
        x1 = w / 4
        x2 = w * 3 / 4

    img = char_image.crop((x1, y1, x2, y2))

    if width != None and rescale:
        img = img.resize((w/2,h/2), resample=Image.BICUBIC)

    return img


def normalize(char_image, factor):
    array = np.array(char_image).astype(np.float32)
    m = np.mean(array)
    s = np.std(array)
    array = np.clip(array, 0.0, 255.0)
    return Image.fromarray(array).convert('L')


def create_char(char_width, char_height, font, char, options={}):
    left_aligned = options.get('left_aligned', False)
    resize_char = options.get('resize_char', False)
    canvas_width = char_width * 2
    canvas_height = char_height * 2
    min_color_delta = options.get('min_color_delta', 32)
    text_color = random.randint(0,255)
    background_color = random_background_color(text_color, min_color_delta=min_color_delta)
    text = char

    image = create_char_background(canvas_width, canvas_height, text_color, background_color, min_color_delta, options=options)
    char_image = Image.new('RGBA', (canvas_width, canvas_height), (0,0,0,0))

    (w,h) = font.calc_text_size(text)
    if left_aligned:
        x = 0.5 * char_width + random.randint(-2,2)
    else:
        x = 0.5 * (canvas_width - w)

    y = 0.5 * (canvas_height - h)

    if random.random() > 0.5:
        text = char_source.random_char() + text
        (w2,h2) = font.calc_text_size(text)
        x -= (w2 - w)

    if random.random() > 0.5:
        text = text + char_source.random_char()

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
    if not left_aligned and not resize_char:
        char_image = perspective_transform(char_image)

    crop_width = w if resize_char else None
    char_image = crop(char_image, crop_width)
    char_image = blur(char_image, options)
    char_image = add_noise(char_image, options)
    return char_image


def create_random_char(options={}):
    font = font_source.random_font(options)
    return create_char(char_width, char_height, font, char_source.random_char())


def CharacterGenerator(batchsize, options={}):
    mean = options.get('mean', None)
    std = options.get('std', None)
    full_alphabet = options.get('full_alphabet', False)
    if full_alphabet:
        char_source = AlphaNumericCharacterSource()
    else:
        char_source = NumericCharacterSource()        
        
    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            font = font_source.random_font(options)
            char = char_source.random_char()
            char_image = create_char(char_width, char_height, font, char, options)
            char_data = np.array(char_image).astype('float32')

            if mean == None:
                mean = np.mean(char_data, axis=(0,1))

            if std == None:
                std = np.std(char_data, axis=(0,1))

            char_data = (char_data - mean) / std

            x.append(char_data.reshape(char_height,char_width,1))
            y.append(char_source.index_for_char(char))

        yield np.array(x),to_categorical(y,char_source.num_chars())


if __name__ == "__main__":
    overview_image = Image.new("L", (char_width * num_char_columns, char_height * num_char_rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)
    options={'min_color_delta':16.0, 'min_blur':0.5, 'max_blur':1.5, 'max_rotation':5.0, 'min_noise':4, 'max_noise':4, 'resize_char':True}
    full_alphabet = options.get('full_alphabet', False)
    if full_alphabet:
        char_source = AlphaNumericCharacterSource()

    for j in range(0,num_char_rows):
        for i in range(0,num_char_columns):
            font = font_source.random_font(options)
            char = char_source.random_char()
            overview_image.paste(create_char(char_width, char_height, font, char, options), (char_width*i, char_height*j))

            #overview_image.paste(Image.fromarray((batch[0][i].reshape(char_height,char_width) * 255).astype('uint8'),mode="L"), (char_width*i, char_height*j))

            if True:
#                print("%02d/%02d: %s" % (j,i, font_tuple[0]))
                overview_draw.text((i * char_width, j * char_height + 10), char)
#                overview_draw.text((i * char_width, j * char_height + 38), "%02d/%02d" % (j,i))

    overview_image.save("overview.png")
