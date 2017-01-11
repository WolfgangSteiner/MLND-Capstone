from CharacterGenerator import create_char_background, rotate, add_noise, blur, crop, perspective_transform
from CharacterGenerator import random_background_color, draw_text
from CharacterGenerator import add_outline, add_shadow, font_source
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

num_char_columns = 2
num_char_rows = 32
debug = True
char_source = NumericCharacterSource()


def create_text_image(image_width = 128, image_height = 32, options={}):
    canvas_width = 640
    canvas_height = 480
    canvas_size = Point(canvas_width, canvas_height)
    font=font_source.random_font(options)
    min_color_delta = options.get('min_color_delta', 32)
    text_color = random.randint(0,255)
    background_color = random_background_color(text_color, min_color_delta=min_color_delta)
    text = char_source.random_char()
    max_noise_scale = options.get('max_noise_scale', 64)

    image = Drawing.create_noise_background(canvas_size, text_color, background_color, min_color_delta, random.uniform(0.5,1.5), max_factor=max_noise_scale)
    char_image = Image.new('RGBA', (canvas_width, canvas_height), (0,0,0,0))

    text = ""
    margin = 4

    for i in range(0,random.randint(1,10)):
        new_text = text + char_source.random_char()
        (w,h) = font.calc_text_size(new_text)
        if w >= canvas_width - 2.0 * margin:
            break
        text = new_text

    (w,h) = font.calc_text_size(text)
    x = 0.5 * (canvas_width - w)
    y = 0.5 * (canvas_height - h)
    x += random_offset(0.5 * (canvas_width - w) - margin)
    y += random_offset(0.5 * (canvas_height - h) - margin)
    y -= font.getoffset(text)[1]

    draw = ImageDraw.Draw(char_image)

    if random.random() > 0.5:
        add_outline(draw, x, y, font, text, text_color)

    draw_text(draw, x, y, text, font, text_color)
    char_image = rotate(char_image, options)

    if random.random() > 0.5:
        shadow_image = add_shadow(char_image, x, y, font, text, text_color)
        image = Image.alpha_composite(image, shadow_image)

    image = Image.alpha_composite(image, char_image)
    image = blur(image, options)
    image = add_noise(image, options)
    return image, text

# #    char_image = perspective_transform(char_image)
#     char_image = crop(char_image, w + margin, rescale=False)
#     return char_image, text


# def CharacterSequenceGenerator(batchsize, options={}):
#     mean = options.get('mean', None)
#     std = options.get('std', None)
#     while True:
#         x = []
#         y = []
#         for i in range(0,batchsize):
#             font_tuple = random_font(options)
#             is_char_border = int(random.random() > 0.5)
#             image = create_segmentation_example(font_tuple, is_char_border, options)
#             image_data = np.array(image).astype('float32')
#
#             if mean == None:
#                 mean = np.mean(image_data, axis=(0,1))
#
#             if std == None:
#                 std = np.std(image_data, axis=(0,1))
#
#             image_data = (image_data - mean) / std
#
#             x.append(image_data.reshape(image_height,image_width,1))
#             y.append(is_char_border)
#
#         yield np.array(x),y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action="store", dest="n", type=int, default=1024)
    parser.add_argument('--max-noise-scale', action="store", dest="max_noise_scale", type=int, default=64)
    parser.add_argument('--max-rotation', action="store", dest="max_rotation", type=float, default=2.5)
    parser.add_argument('--directory', action='store', dest='data_dir', default='data')
    parser.add_argument("--save", help="save image as png along with a pickle of the labels", action="store_true")
    args = parser.parse_args()

    image_width = 256
    image_height = 32
    options={'min_color_delta':16.0, 'min_blur':0.5, 'max_blur':1.5, 'max_rotation':args.max_rotation, 'min_noise':4, 'max_noise':16, 'add_background_lines':False}
    options['max_size'] = 8.0
    options['min_size'] = 0.75
    options['max_noise_scale'] = args.max_noise_scale

    if args.save:
        labels = {}
        mkdir(args.data_dir)
        for i in range(args.n):
            sys.stdout.write("\r%d" % (i+1))
            sys.stdout.flush()
            id = str(uuid.uuid4())
            char_image, label = create_text_image(image_width, image_height, options)
            labels[id] = label
            char_image.save(args.data_dir + "/" + id + ".png")
        file = open(args.data_dir + '/' + 'labels.pickle', 'wb')
        print
        print ("Writing labels.pickle ...")
        pickle.dump(labels, file, -1)

    else:
        char_image, label = create_text_image(image_width, image_height, options)
        char_image.save("overview.png")
