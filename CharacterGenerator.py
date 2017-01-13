#! /usr/bin/python2
from __future__ import print_function
from PIL import Image, ImageDraw
import random
import numpy as np
import math
from keras.utils.np_utils import to_categorical
from CharacterSource import NumericCharacterSource, AlphaNumericCharacterSource
from FontSource import FontSource
import Drawing
from Point import Point
import Utils
import pickle


char_height = 32
char_width = 32
char_size = Point(char_height, char_width)
canvas_width = 2 * char_width
canvas_height = 2 * char_height
num_char_columns = 16
num_char_rows = 32
debug = False
char_source = NumericCharacterSource()
font_source = FontSource()


def create_char(char_width, char_height, font, char, options={}):
    min_color_delta = options.get('min_color_delta', 32)
    canvas_width = char_width * 2
    canvas_height = char_height * 2
    canvas_size = 2 * char_size
    text_color = random.randint(0,255)
    background_color = Drawing.random_background_color(text_color, min_color_delta=min_color_delta)

    image = Drawing.create_noise_background(canvas_size, text_color, background_color, min_color_delta, random.uniform(0.5,1.5), max_factor=8)
    char_img = Image.new('RGBA', (canvas_width, canvas_height), (0,0,0,0))

    text = char
    (w,h) = font.calc_text_size(text)

    x = 0.5 * (canvas_width - w)
    y = 0.5 * (canvas_height - h)

    if random.random() > 0.5:
        text = char_source.random_char() + text
        (w2,h2) = font.calc_text_size(text)
        x -= (w2 - w)

    if random.random() > 0.5:
        text = text + char_source.random_char()

    x += random.randint(-2,2)
    y += (random.random() - 0.5) * (char_height - h)

    draw = ImageDraw.Draw(char_img)
    Drawing.draw_text_with_random_outline(draw, x, y, text, font, text_color)

    if random.random() > 0.5:
        image = Drawing.add_shadow(char_img, image, x, y, font, text, text_color)

    char_img = Image.alpha_composite(image, char_img)
    char_img = Drawing.random_rotate(char_img, options)
    #char_img = Drawing.perspective_transform(char_img)

    crop_width = w + random.randint(-2,2)
    char_img = Drawing.crop(char_img, crop_width)
    char_img = Drawing.random_blur(char_img, options)
    char_img = Drawing.add_noise(char_img, options)
    return char_img


def CharacterGenerator(batchsize, options={}):
    full_alphabet = options.get('full_alphabet', False)
    if full_alphabet:
        char_source = AlphaNumericCharacterSource()
    else:
        char_source = NumericCharacterSource()

    show_progress = options.get('show_progress', False)

    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            if show_progress:
                Utils.progress_bar(i+1, batchsize)
            font = font_source.random_font(options)
            char = char_source.random_char()
            char_img = create_char(char_width, char_height, font, char, options)
            char_data = np.array(char_img).astype('float32') / 255.0

            x.append(char_data.reshape(char_height,char_width,1))
            y.append(char_source.index_for_char(char))

        yield np.array(x),to_categorical(y,char_source.num_chars())


def generate_test_data(file_name, num_examples, options={}):
    options['show_progress'] = True
    gen = CharacterGenerator(num_examples, options)
    X,y = gen.next()
    with open(file_name, "wb") as f:
        pickle.dump(X,f)
        pickle.dump(y,f)


if __name__ == "__main__":
    overview_image = Image.new("L", (char_width * num_char_columns, char_height * num_char_rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)
    options={'min_color_delta':16, 'min_blur':1.5, 'max_blur':3.5, 'max_rotation':15.0, 'min_noise':4, 'max_noise':8, 'resize_char':True, 'full_alphabet':False, 'add_background_lines':True}
    full_alphabet = options.get('full_alphabet', False)
    if full_alphabet:
        char_source = AlphaNumericCharacterSource()

    for j in range(0,num_char_rows):
        for i in range(0,num_char_columns):
            font = font_source.random_font(options)
            char = char_source.random_char()
            overview_image.paste(create_char(char_width, char_height, font, char, options), (char_width*i, char_height*j))
            overview_draw.text((i * char_width, j * char_height + 10), char)

    overview_image.save("overview.png")
