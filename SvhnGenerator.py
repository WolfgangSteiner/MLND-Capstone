from Rectangle import Rectangle
from PIL import Image, ImageDraw
import pickle
from Point import Point
import random
import Drawing
import numpy as np

detector_size = Point(32,32)
img_path = "train"

def read_dict(pickle_file):
    f = open(pickle_file, "rb")
    return pickle.load(f)
    f.close()

bboxes = read_dict(img_path + '/' + "bboxes.pickle")

def calc_global_bbox(bbox_array):
    x1 = y1 = 1e6
    x2 = y2 = -1e6

    for bbox in bbox_array:
        x1 = min(x1, bbox[0])
        y1 = min(y1, bbox[1])
        x2 = max(x2, bbox[0] + bbox[2])
        y2 = max(y2, bbox[1] + bbox[3])

    return Rectangle(x1,y1,x2,y2)


def generate_random_example():
    id = random.choice(bboxes.keys())
    png_file = img_path + '/' + id + '.png'
    return generate_example(png_file, bboxes[id])


def does_rect_intersect_char(rect, bbox_array, scale_factor):
    for bbox in bbox_array:
        r = Rectangle.from_point_and_size(Point(bbox[0],bbox[1]), Point(bbox[2],bbox[3]))
        r *= scale_factor
        if r.calc_overlap(rect) > 0.85:
            return True

    return False


def generate_example(png_file, bbox_array):
    img = Image.open(png_file).convert('L')
    global_bbox = calc_global_bbox(bbox_array)
    bbox_height = global_bbox.height()
    factor = float(detector_size.y) / bbox_height if bbox_height > 0 else 1.0
    global_bbox *= factor
    img = Drawing.scale_image(img, factor)
    img_size = Point(img.size[0], img.size[1])
    offset = Point.random(img_size - detector_size)
    crop_rect = Rectangle.from_point_and_size(offset, detector_size)
    label = does_rect_intersect_char(crop_rect, bbox_array, factor)
    return img.crop(crop_rect.as_array()), int(label)


def SvhnDetectionGenerator(batchsize, options={}):
    while True:
        x = []
        y = []
        for i in range(0,batchsize):
            img, label = generate_random_example()
            img_data = np.array(img).astype('float32')
            x.append(img_data.reshape(detector_size.y,detector_size.x,1))
            y.append(label)

        yield np.array(x),y


if __name__ == '__main__':
    num_char_columns = 32
    num_char_rows = 32

    overview_image = Image.new("RGB", (detector_size.x * num_char_columns, detector_size.y * num_char_rows), (0,0,0))
    overview_draw = ImageDraw.Draw(overview_image)

    for j in range(0,num_char_rows):
        for i in range(0,num_char_columns):
            img, label = generate_random_example()
            overview_image.paste(img, (detector_size.x*i, detector_size.y*j))
            overview_draw.text((i * detector_size.x, j * detector_size.y + 20), str(label), fill=(0,255,0))

    overview_image.save("overview.png")
