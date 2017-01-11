from CharacterGenerator import font_source
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform, ImageChops
import random
import numpy as np
import uuid
import argparse
import sys, errno, os
import pickle
from CharacterSource import NumericCharacterSource
from FontSource import FontSource
from Utils import mkdir
from MathUtils import random_offset
from Point import Point
import Drawing
import shutil
import Utils

num_char_columns = 2
num_char_rows = 32
debug = True
char_source = NumericCharacterSource()


def create_text_image(image_width = 640, image_height = 480, options={}):
    image_width = 640
    image_height = 480
    image_size = Point(image_width, image_height)
    font=font_source.random_font(options)
    min_color_delta = options.get('min_color_delta', 32)
    text_color = random.randint(0,255)
    background_color = Drawing.random_background_color(text_color, min_color_delta=min_color_delta)
    text = char_source.random_char()
    max_noise_scale = options.get('max_noise_scale', 64)

    image = Drawing.create_noise_background(image_size, text_color, background_color, min_color_delta, random.uniform(0.5,1.5), max_factor=max_noise_scale)
    char_image = Image.new('RGBA', (image_width, image_height), (0,0,0,0))

    text = ""
    margin = 4

    for i in range(0,random.randint(1,10)):
        new_text = text + char_source.random_char()
        (w,h) = font.calc_text_size(new_text)
        if w >= image_width - 2.0 * margin:
            break
        text = new_text

    (w,h) = font.calc_text_size(text)
    x = 0.5 * (image_width - w)
    y = 0.5 * (image_height - h)
    x += random_offset(0.5 * (image_width - w) - margin)
    y += random_offset(0.5 * (image_height - h) - margin)
    y -= font.getoffset(text)[1]

    draw = ImageDraw.Draw(char_image)
    Drawing.draw_text_with_random_outline(draw, x, y, text, font, text_color)
    char_image = Drawing.random_rotate(char_image, options)

    if random.random() > 0.5:
        image = Drawing.add_shadow(char_image, image, x, y, font, text, text_color)

    image = Image.alpha_composite(image, char_image)
    image = Drawing.random_blur(image, options)
    image = Drawing.add_noise(image, options)
    return image, text


def create_test_images(dir, max_rotation=2.5, max_noise_scale=1, n=100):
    options = \
    { \
        'min_color_delta':16.0, \
        'min_blur':0.5, \
        'max_blur':1.5, \
        'max_rotation':max_rotation, \
        'min_noise':4, \
        'max_noise':16, \
        'min_size': 0.75, \
        'max_size': 8.0, \
        'max_noise_scale': max_noise_scale \
    }

    Utils.rmdir(dir)
    Utils.mkdir(dir)
    labels = {}
    for i in range(n):
        Utils.progress_bar(i+1,n, message=dir)
        id = str(uuid.uuid4())
        char_image, label = create_text_image(640, 480, options)
        labels[id] = label
        char_image.save(dir + "/" + id + ".png")

    with open(dir + '/' + 'labels.pickle', 'wb') as f:
        print
        print ("Writing labels.pickle ...")
        pickle.dump(labels, f, -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action="store", dest="n", type=int, default=1024)
    parser.add_argument('--max-noise-scale', action="store", dest="max_noise_scale", type=int, default=64)
    parser.add_argument('--max-rotation', action="store", dest="max_rotation", type=float, default=2.5)
    parser.add_argument('--directory', action='store', dest='data_dir', default='data')
    args = parser.parse_args()
    create_images(args.data_dir, args.max_rotation, args.max_noise_scale, args.n)
