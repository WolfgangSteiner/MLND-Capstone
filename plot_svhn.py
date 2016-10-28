#! /usr/bin/python2
from __future__ import print_function
from PIL import Image, ImageDraw
import random
import glob, os, os.path
import numpy as np
import re
import math
from scipy.io import loadmat


char_height = 32
char_width = 32
num_char_columns = 32
num_char_rows = 32

overview_image = Image.new("L", (char_width * num_char_columns, char_height * num_char_rows), 255)
draw = ImageDraw.Draw(overview_image)

def load_svhn(file_name):
    mat = loadmat(file_name)
    X = mat['X'].astype(np.float32)
    n = X.shape[0]
    X = X.transpose((3,0,1,2)).reshape(-1,32,32,3)
    X = np.mean(X, axis=3)

    y = mat['y']
    y[y==10] = 0

    return X,y


X,y = load_svhn("test_32x32.mat")

for j in range(0,num_char_rows):
    for i in range(0,num_char_columns):
        idx = random.randint(0,X.shape[0]-1)
        img = Image.fromarray(X[idx]).convert('L')
        label = y[idx][0]
        overview_image.paste(img, (char_width*i, char_height*j))
        draw.text((i * char_width, j * char_height + 20), str(label))

            #overview_image.paste(Image.fromarray((batch[0][i].reshape(char_height,char_width) * 255).astype('uint8'),mode="L"), (char_width*i, char_height*j))

overview_image.save("test_32x32.png")
