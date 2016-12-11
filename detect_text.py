#from CharacterSequenceGenerator import create_char_sequence
import numpy as np
import pickle, sys, argparse, glob, pickle
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform, ImageChops
from collections import namedtuple
from segmentation import predict_word
import argparse

def quantize(a, q):
    return int(a/q) * q

char_detector = load_model("detection008.hdf5")
Size = namedtuple('Size', 'w h')
Pos = namedtuple('Pos', 'x y')

detector_size = Size(32,32)
detector_overlap = 2

def rescale_image(img, scale_factor):
    (w,h) = img.size
    new_w = quantize(w * scale_factor,detector_size.w)
    new_h = quantize(h * scale_factor,detector_size.h)
    return img.resize((new_w, new_h), resample=Image.BICUBIC), (float(new_w) / w, float(new_h) / h)


def prepare_image_for_classification(image):
    w,h = image.size
    image_data = np.array(image).astype('float32')
    return image_data.reshape(h,w,1)


def make_rect(pos, size):
    return [pos.x, pos.y, pos.x + size.w, pos.y + size.h]


def unscale_rect(rect, factors):
    return [rect[0]/factors[0], rect[1]/factors[1], rect[2]/factors[0], rect[3]/factors[1]]


def check_text(img, pos):
    window_rect = make_rect(pos, detector_size)
    window = img.crop(window_rect)
    window_data = prepare_image_for_classification(window)
    is_text = char_detector.predict(window_data)[0] > 0.75
    return is_text, window_rect


def detect_text(img):
    y = 0
    x = 0
    (w,h) = img.size
    data = []
    while y < h:
        x = 0
        while x < w:
            window_rect = make_rect(Pos(x,y), detector_size)
            window = img.crop(window_rect)
            window_data = prepare_image_for_classification(window)
            data.append(window_data.reshape(detector_size.h,detector_size.w,1))
            x += detector_size.w / detector_overlap
        y += detector_size.h / detector_overlap

    result = char_detector.predict(np.array(data))
    return result


def scan_image_at_scale(img, scale_factor):
    img, scale_factors = rescale_image(img, scale_factor)
    (w,h) = img.size
    y = 0
    result_array = []
    is_text_vector = detect_text(img)
    i = 0

    while y < h:
        x1 = 0
        x2 = 0
        is_in_word = False
        x = 0
        while x < w:
            window_rect = make_rect(Pos(x,y), detector_size)
            is_text = is_text_vector[i] > 0.25
            if is_text:
                if not is_in_word:
                    is_in_word = True
                    x1 = x
                    x2 = x1

                x2 = x + detector_size.w
                is_in_word = True
            elif is_in_word and x > x2 + detector_size.w + 1:
                y2 = y + detector_size.h
                word_rect = [x1, y, x2, y2]
                is_in_word = False
                text = predict_word(img.crop(word_rect))
                scaled_word_rect = unscale_rect(word_rect, scale_factors)
                result_array.append((scaled_word_rect, text))

            x += detector_size.w / detector_overlap
            i += 1
        y += detector_size.h / detector_overlap

    return result_array


def scan_image(img, max_factor=1.0, min_factor=None):
    factor = max_factor
    result_array = []
    (w,h) = img.size

    if min_factor == None:
        min_size = detector_size
    else:
        min_size = Size(max(min_factor * w, detector_size.w), max(min_factor * h, detector_size.h))

    while img.size[0] * factor > min_size.w and img.size[1] * factor > min_size.h:
        result_array += scan_image_at_scale(img, factor)
        factor *= 0.75

    return result_array


def draw_detected_text(img, result_array):
    draw = ImageDraw.Draw(img)

    for rect, text in result_array:
        draw.rectangle(rect, outline=(0,255,0))
        draw.text([rect[0],rect[3]], text, fill=(0,255,0))


def scan_image_file(file_path):
    img = Image.open(file_path)
    result_array = scan_image(img, 0.75, 0.25)
    result_img = img.convert('RGB')
    draw_detected_text(result_img, result_array)
    result_img.show()


def test_image_file(file_path):
    img = Image.open(file_path)
    result_array = scan_image(img, 0.75, 0.25)
    if len(result_array):
        return result_array[0][1]
    else:
        return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action="store", dest="n", type=int, default=1024)
    parser.add_argument('--directory', action='store', dest='data_dir', default='data')
    args = parser.parse_args()

    try:
        f = open(args.data_dir + '/labels.pickle', 'rb')
        labels = pickle.load(f)
    except:
        raise IOError

    n = 0
    true_positives = 0

    for id, label in labels.iteritems():
        predicted_label = test_image_file(args.data_dir + "/" + id + ".png")
        if predicted_label == label:
            true_positives += 1
        n += 1
        sys.stdout.write("\r%d/%d: accuracy = %.2f%%" % (n, len(labels), float(true_positives)/n)*100.0)
        sys.stdout.flush()

    print
