from CharacterGenerator import font_source
from PIL import Image, ImageDraw
import random
import numpy as np
import uuid
import argparse
import sys, errno, os
from CharacterSource import NumericCharacterSource
from FontSource import FontSource
from Utils import mkdir
from MathUtils import random_offset
from Rectangle import Rectangle
from Point import Point
import Drawing


num_char_columns = 2
num_char_rows = 32
debug = True
char_source = NumericCharacterSource()


def create_text_detection_example(options={}):
    canvas_width = 640
    canvas_height = 480
    canvas_size = Point(canvas_width, canvas_height)
    font=font_source.random_font(options)
    min_color_delta = options.get('min_color_delta', 32)
    text_color = random.randint(0,255)
    background_color = random_background_color(text_color, min_color_delta=min_color_delta)
    text = char_source.random_char()


    image = Drawing.create_noise_background(canvas_size, text_color, background_color, min_color_delta, random.uniform(0.5,1.5), max_factor=64)
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
    Drawing.draw_text_with_random_outline(draw, x, y, text, font, text_color)

    if random.random() > 0.5:
        image = Drawing.add_shadow(char_image, image, x, y, font, text, text_color)

    image = Image.alpha_composite(image, char_image)
    image = Drawing.random_blur(image, options)
    image = Drawing.add_noise(image, options)
    image = image.resize((256, 256), resample=Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    scale_factor = Point(256.0 / canvas_width, 256.0 / canvas_height)
    text_rect = Rectangle.from_point_and_size(Point(x,y + font.getoffset(text)[1]), Point(w,h)).scale(scale_factor)

    labels = []
    for j in range(0,32):
        for i in range(0,32):
            window = Rectangle.from_point_and_size(Point(i,j) * 8, Point(16,16))
            if window.calc_overlap(text_rect) > 0.5:
                labels[j*32 + i] = 1
            else:
                lables[j*32 + i] = 0

    return image, labels


def TextDetectionGenerator(batchsize, options={}):
    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            font_tuple = random_font(options)
            image,labels = create_segmentation_example(font_tuple, is_char_border, options)
            image_data = np.array(image).astype('float32')
            x.append(image_data.reshape(image_height,image_width,1))
            y.append(labels)

        yield np.array(x),np.array(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action="store", dest="n", type=int, default=1024)
    parser.add_argument('--directory', action='store', dest='data_dir', default='data')
    parser.add_argument("--save", help="save image as png along with a pickle of the labels", action="store_true")
    args = parser.parse_args()

    image_width = 256
    image_height = 32
    options={'min_color_delta':16.0, 'min_blur':0.5, 'max_blur':1.5, 'max_rotation':2.0, 'min_noise':4, 'max_noise':4, 'add_background_lines':False}
    options['max_size'] = 8.0
    options['min_size'] = 0.75

    if args.save:
        labels = {}
        mkdir(args.data_dir)
        for i in range(args.n):
            sys.stdout.write("\r%d" % (i+1))
            sys.stdout.flush()
            id = str(uuid.uuid4())
            char_image, label = create_text_detection_example(options)
            labels[id] = label
            char_image.save(args.data_dir + "/" + id + ".png")
        file = open(args.data_dir + '/' + 'labels.pickle', 'wb')
        print
        print ("Writing labels.pickle ...")
        pickle.dump(labels, file, -1)

    else:
        char_image, label = create_text_detection_example(options)
        char_image.save("overview.png")
