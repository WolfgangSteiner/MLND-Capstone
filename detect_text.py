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
import os, shutil
import Utils
from MathUtils import levenshtein_distance
import Levenshtein

def quantize(a, q):
    return int(a/q) * q

char_detector = load_model("models/detection.hdf5")

detector_size = Point(32,32)
detector_overlap = 4
detector_scaling_factor = 0.75
detector_scaling_min = 0.1
detector_scaling_max = 1.0
detector_threshold = 0.85


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
    image_data = np.array(image).astype('float32')/255.0
#    print "image size: %d, %d" % (w,h), image_data.shape
    return image_data.reshape(h,w,1)


def check_text(img, pos):
    window_rect = Rectangle.from_point_and_size(pos, detector_size)
    window = img.crop(window_rect.as_array)
    window_data = prepare_image_for_classification(window)
    is_text = char_detector.predict(window_data)[0] > text_detector_threshold
    return is_text, window_rect


def detect_text(img, detector_overlap=2):
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


def scan_image_at_scale(img, scale_factor, rect_array, detector_overlap=2, detector_threshold=0.85):
    scaled_img, scale_factors = rescale_image(img, scale_factor)
    (w,h) = scaled_img.size
    y = 0
    delta_x = detector_size.x / detector_overlap
    delta_y = detector_size.y / detector_overlap
    result_array = []
    is_text_vector = detect_text(scaled_img, detector_overlap)
    i = 0

    while y <= h - detector_size.y:
        x = 0
        while x <= w - detector_size.x:
            window_rect = Rectangle.from_point_and_size(Point(x,y), detector_size)
            is_text = is_text_vector[i] > detector_threshold
            if is_text:
                rect_array.add(window_rect.unscale(scale_factors))

            x += delta_x
            i += 1
        y += delta_y


def infer_text(img, rect):
    text_line_img = img.crop(rect.as_array())
    text_line_img = rescale_image_to_height(text_line_img, detector_size.y)
    text, seg_array = predict_word(text_line_img)
    return text, seg_array


def scan_image(img, max_factor=1.0, min_factor=None, detector_scaling_factor=0.5, detector_overlap=2, detector_threshold=0.85):
    rect_array = RectangleArray()
    factor = max_factor
    result_array = []
    (w,h) = img.size

    if min_factor == None:
        min_size = detector_size
    else:
        min_size = Point(max(min_factor * w, detector_size.x), max(min_factor * h, detector_size.y))

    while img.size[0] * factor >= min_size.x and img.size[1] * factor >= min_size.y:
        scan_image_at_scale(img, factor, rect_array, detector_overlap, detector_threshold)
        factor *= detector_scaling_factor

    rect_array.finalize()
    for r in rect_array.list:
        #shrink the bounding box as it is overestimated in vertical direction
        #r = r.shrink_with_factor(Point(1.0, 0.75))
        text, seg_array = infer_text(img, r)
        if len(text):
            result_array.append((r,text,seg_array))

    return result_array, rect_array


def draw_detected_text(img, result_array, label):
    draw = ImageDraw.Draw(img)
    for rect, text, seg_array in result_array:
        draw.rectangle(rect.as_array(), outline=(0,255,0))
        text_color = (0,255,0) if text == label else (255,0,0)
        draw.text([rect.x1,rect.y1], text, fill=text_color)


def draw_segmentation(img, result_array):
    draw = ImageDraw.Draw(img)
    for rect, text, seg_array in result_array:
        factor = 32.0 / rect.height()
        for s in seg_array:
            x = int(rect.x1 + s / factor)
            draw.line((x, rect.y1, x, rect.y2), fill=(128,128,255), width=2)


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
    result_array, rect_array = scan_image(img, detector_scaling_max, detector_scaling_min, detector_scaling_factor, detector_overlap)
    result_img = img.convert('RGB')
    draw_separate_candidates(result_img, rect_array)
    draw_detected_text(result_img, result_array)
    draw_segmentation(result_img, result_array)
    result_img.show()


def has_correct_label(result_array, label):
    for _, text, _ in result_array:
        if text == label:
            return True
    return False


def test_image_file(file_path, label):
    img = Image.open(file_path)
    output_dir = os.path.dirname(file_path) + "/output"
    correct_dir = output_dir + "/correct"
    incorrect_dir = output_dir + "/incorrect"
    Utils.mkdir(correct_dir)
    Utils.mkdir(incorrect_dir)

    #img = scale_image(img, 2.0)
    if img.mode != 'L':
        img = img.convert('L')
    result_array, rect_array = scan_image(img, detector_scaling_max, detector_scaling_min, detector_scaling_factor, detector_overlap, detector_threshold)
    result_img = img.convert('RGB')
    draw_separate_candidates(result_img, rect_array)
    draw_bounding_boxes(result_img, rect_array)
    draw_detected_text(result_img, result_array, label)
    draw_segmentation(result_img, result_array)
    result_img.save("result.png")
    Utils.display_image("result.png")
    result_path = correct_dir if has_correct_label(result_array, label) else incorrect_dir
    result_path += '/' + os.path.basename(file_path)
    result_img.save(result_path)
    return result_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', action="store", dest="n", type=int, default=1024)
    parser.add_argument('--directory', action='store', dest='data_dir', default='data')
    args = parser.parse_args()
    output_dir = args.data_dir + "/output"
    Utils.rmdir(output_dir)

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
        result_array = test_image_file(args.data_dir + "/" + id + ".png", label)

        predicted_label = result_array[0][1] if len(result_array) else ""
        predicted_text = ""

        for r in result_array:
            predicted_text += r[1] + " "

        num_digits += len(label)
        distance = Levenshtein.distance(label, predicted_label)
        num_correct_digits += len(label) - min(distance,len(label))

        n += 1
        if n > args.n:
            break

        accuracy = float(num_correct_digits)/num_digits
        print "%d/%d: %s %s -> %s accuracy = %.4f" % (n, min(len(labels),args.n), id[0:6], label, predicted_text, accuracy)
    print

    end_time = timer()

    print "Finished in %fs" % (end_time - start_time)
