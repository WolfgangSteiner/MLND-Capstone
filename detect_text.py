#from CharacterSequenceGenerator import create_char_sequence
import numpy as np
import pickle, sys, argparse, glob, pickle
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform, ImageChops
from collections import namedtuple
from segmentation import predict_word
import argparse
from RectangleArray import RectangleArray
from Rectangle import Rectangle
from Point import Point
from timeit import default_timer as timer
from Drawing import scale_image
import os
from Utils import mkdir
from MathUtils import levenshtein_distance

def quantize(a, q):
    return int(a/q) * q

#char_detector = load_model("detection012-svhn-2.hdf5")
char_detector = load_model("detection013.hdf5")

detector_size = Point(32,32)
detector_overlap = 4
detector_scaling_factor = 0.5
detector_scaling_min = 0.5
detector_scaling_max = 2.0
text_detector_threshold = 0.75


def rescale_image(img, scale_factor):
    (w,h) = img.size
    new_w = quantize(w * scale_factor,detector_size.x)
    new_h = quantize(h * scale_factor,detector_size.y)
    return img.resize((new_w, new_h), resample=Image.BILINEAR), Point(float(new_w) / w, float(new_h) / h)


def rescale_image_to_height(img, height):
    (w,h) = img.size
    factor = float(height) / h
    new_w = int(w * factor)
    return img.resize((new_w, height), resample=Image.BILINEAR)


def prepare_image_for_classification(image):
    w,h = image.size
    image_data = np.array(image).astype('float32')
#    print "image size: %d, %d" % (w,h), image_data.shape
    return image_data.reshape(h,w,1)


def check_text(img, pos):
    window_rect = Rectangle.from_point_and_size(pos, detector_size)
    window = img.crop(window_rect.as_array)
    window_data = prepare_image_for_classification(window)
    is_text = char_detector.predict(window_data)[0] > text_detector_threshold
    return is_text, window_rect


def detect_text(img):
    y = 0
    (w,h) = img.size
    delta_x = detector_size.x / detector_overlap
    delta_y = detector_size.y / detector_overlap

    data = []
    while y <= h - detector_size.y:
        x = 0
        while x <= w - detector_size.x:
            window_rect = Rectangle.from_point_and_size(Point(x,y), detector_size)
            window = img.crop(window_rect.as_array())
            window_data = prepare_image_for_classification(window)
            data.append(window_data.reshape(detector_size.y,detector_size.x,1))
            x += delta_x
        y += delta_y

    result = char_detector.predict(np.array(data))
    return result


def scan_image_at_scale(img, scale_factor, rect_array):
    scaled_img, scale_factors = rescale_image(img, scale_factor)
    (w,h) = scaled_img.size
    y = 0
    delta_x = detector_size.x / detector_overlap
    delta_y = detector_size.y / detector_overlap
    result_array = []
    is_text_vector = detect_text(scaled_img)
    i = 0

    while y <= h - detector_size.y:
        x = 0
        while x <= w - detector_size.x:
            window_rect = Rectangle.from_point_and_size(Point(x,y), detector_size)
            is_text = is_text_vector[i] > text_detector_threshold
            if is_text:
                rect_array.add(window_rect.unscale(scale_factors))

            x += delta_x
            i += 1
        y += delta_y


def infer_text(img, rect):
    text_line_img = img.crop(rect.as_array())
    text_line_img = rescale_image_to_height(text_line_img, detector_size.y)
    text = predict_word(text_line_img)
    return text


def scan_image(img, max_factor=1.0, min_factor=None):
    rect_array = RectangleArray()
    factor = max_factor
    result_array = []
    (w,h) = img.size

    if min_factor == None:
        min_size = detector_size
    else:
        min_size = Point(max(min_factor * w, detector_size.x), max(min_factor * h, detector_size.y))

    while img.size[0] * factor >= min_size.x and img.size[1] * factor >= min_size.y:
        scan_image_at_scale(img, factor, rect_array)
        factor *= detector_scaling_factor

    rect_array.finalize()
    for r in rect_array.list:
        text = infer_text(img, r)
        if len(text):
            result_array.append((r,text))

    return result_array, rect_array


def draw_detected_text(img, result_array):
    draw = ImageDraw.Draw(img)
    for rect, text in result_array:
        draw.rectangle(rect.as_array(), outline=(0,255,0))
        draw.text([rect.x1,rect.y1], text, fill=(0,255,0))


def draw_separate_candidates(img, rect_array):
    draw = ImageDraw.Draw(img)
    for rect in rect_array.separate_list:
        draw.rectangle(rect.as_array(), outline=(128,0,0))


def draw_bounding_boxes(img, rect_array):
    draw = ImageDraw.Draw(img)
    for rect in rect_array.list:
        draw.rectangle(rect.as_array(), outline=(0,0,255))


def scan_image_file(file_path):
    img = Image.open(file_path)
    result_array, rect_array = scan_image(img, detector_scaling_max, detector_scaling_min)
    result_img = img.convert('RGB')
    draw_separate_candidates(result_img, rect_array)
    draw_detected_text(result_img, result_array)
    result_img.show()


def test_image_file(file_path):
    img = Image.open(file_path)
    output_dir = os.path.dirname(file_path) + "/output"
    mkdir(output_dir)
    result_path = output_dir + '/' + os.path.basename(file_path)

    #img = scale_image(img, 2.0)
    if img.mode != 'L':
        img = img.convert('L')
    result_array, rect_array = scan_image(img, detector_scaling_max, detector_scaling_min)
    result_img = img.convert('RGB')
    draw_separate_candidates(result_img, rect_array)
    draw_bounding_boxes(result_img, rect_array)
    draw_detected_text(result_img, result_array)
    result_img.save("result.png")
    result_img.save(result_path)
    return result_array


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
    num_digits = 0
    num_correct_digits = 0

    start_time = timer()

    for id, label in labels.iteritems():
        result_array = test_image_file(args.data_dir + "/" + id + ".png")

        predicted_label = result_array[0][1] if len(result_array) else ""
        predicted_text = ""

        for r in result_array:
            predicted_text += r[1] + " "

        num_digits += len(label)
        distance = levenshtein_distance(label, predicted_label)
        num_correct_digits += len(label) - min(distance,len(label))

        n += 1
        if n > args.n:
            break

        accuracy = float(num_correct_digits)/num_digits
        print "%d/%d: %s %s -> %s accuracy = %.4f" % (n, min(len(labels),args.n), id[0:6], label, predicted_text, accuracy)
    print

    end_time = timer()

    print "Finished in %fs" % (end_time - start_time)
