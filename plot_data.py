#! /usr/bin/python2
from __future__ import print_function
from PIL import Image, ImageDraw
import numpy as np
import math
from Point import Point
import Utils
import pickle


def plot_data(data_file, rows=32, cols=32):
    print("loading test data...")
    with open(data_file) as f:
        X = pickle.load(f)
        y = pickle.load(f)

    char_height = X.shape[1]
    char_width = X.shape[2]

    overview_image = Image.new("L", (char_width * cols, char_height * rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)
    for j in range(0,rows):
        for i in range(0,cols):
            img = Image.fromarray(X[j*cols+i].reshape((char_height, char_width))*255.0)
            overview_image.paste(img, (char_width*i, char_height*j))
            label = y[j*cols+i].argmax() if len(y.shape) > 1 else y[j*cols + i]
            overview_draw.text((i * char_width, j * char_height + 20), str(label))

    overview_image.save(data_file.replace(".pickle", ".png"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data')
    parser.add_argument('--cols', action="store", dest="cols", type=int, default=32)
    parser.add_argument('--rows', action="store", dest="rows", type=int, default=32)
    args = parser.parse_args()
    plot_data(args.test_data, args.rows, args.cols)
