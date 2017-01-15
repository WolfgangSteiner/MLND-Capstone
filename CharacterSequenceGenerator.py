from CharacterGenerator import font_source
from CharacterSource import NumericCharacterSource, AlphaNumericCharacterSource
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform, ImageChops
from Utils import mkdir
import random
import numpy as np
import uuid
import argparse
import sys, errno, os
import pickle
import Drawing
import Utils

num_char_columns = 2
num_char_rows = 32
debug = True
char_source = NumericCharacterSource()

def create_char_sequence(image_width = 128, image_height = 32, options={}):
    canvas_width = image_width * 2
    canvas_height = image_height * 2
    font = font_source.random_font(options)
    min_color_delta = options.get('min_color_delta', 32)
    text_color = random.randint(0,255)
    background_color = Drawing.random_background_color(text_color, min_color_delta=min_color_delta)
    text = char_source.random_char()

    image = Drawing.create_char_background(canvas_width, canvas_height, text_color, background_color, min_color_delta, options=options)
    char_image = Image.new('RGBA', (canvas_width, canvas_height), (0,0,0,0))

    text = ""

    for i in range(0,random.randint(1,10)):
        text += char_source.random_char()

    (w,h) = font.calc_text_size(text)
    x = 0.5 * (canvas_width - w)
    y = 0.5 * (canvas_height - h)
    margin = random.random() * 16
    x += (random.random() - 0.5) * 0.5 * margin
    y += (random.random() - 0.5) * (image_height - h)

    draw = ImageDraw.Draw(char_image)
    Drawing.draw_text_with_random_outline(draw, x, y, text, font, text_color)

    if random.random() > 0.5:
        image = Drawing.add_shadow(char_image, image, x, y, font, text, text_color)

    char_image = Image.alpha_composite(image, char_image)
    char_image = Drawing.random_rotate(char_image, options)
#    char_image = perspective_transform(char_image)
    char_image = Drawing.crop(char_image, w + margin, rescale=False)
    char_image = Drawing.random_blur(char_image, options)
    char_image = Drawing.add_noise(char_image, options)
    return char_image, text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action="store", dest="n", type=int, default=1024)
    parser.add_argument('--directory', action='store', dest='data_dir', default='data')
    parser.add_argument("--save", help="save image as png along with a pickle of the labels", action="store_true")
    args = parser.parse_args()

    image_width = 256
    image_height = 32
    options={'min_color_delta':16.0, 'min_blur':0.5, 'max_blur':1.5, 'max_rotation':2.0, 'min_noise':4, 'max_noise':4, 'add_background_lines':False}
    options['full_alphabet'] = False

    full_alphabet = options.get('full_alphabet', False)
    if full_alphabet:
        char_source = AlphaNumericCharacterSource()

    if args.save:
        labels = {}
        mkdir(args.data_dir)
        for i in range(args.n):
            Utils.progress_bar(i+1, args.n)
            id = str(uuid.uuid4())
            char_image, label = create_char_sequence(image_width, image_height, options)
            labels[id] = label
            char_image.save(args.data_dir + "/" + id + ".png")
        file = open(args.data_dir + '/' + 'labels.pickle', 'wb')
        print
        print ("Writing labels.pickle ...")
        pickle.dump(labels, file, -1)

    else:
        overview_image = Image.new("L", (image_width * num_char_columns, image_height * num_char_rows), 255)
        overview_draw = ImageDraw.Draw(overview_image)
        for j in range(0,num_char_rows):
            for i in range(0,num_char_columns):
                char_image, label = create_char_sequence(image_width, image_height, options)
                overview_image.paste(char_image, (image_width*i, image_height*j))

        overview_image.save("overview.png")
