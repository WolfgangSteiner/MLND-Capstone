import numpy as np
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from Common import load_svhn
from sys import argv
import pickle
from PIL import Image,ImageDraw
import argparse


def open_image(img_name):
    return Image.open(args.data_dir + '/' + img_name + ".png")


parser = argparse.ArgumentParser()
parser.add_argument('-n', action="store", dest="n", type=int, default=1024)
parser.add_argument('--num-cols', action="store", dest="char_cols", type=int, default=32)
parser.add_argument('--num-rows', action="store", dest="char_rows", type=int, default=32)
parser.add_argument('--directory', action='store', dest='data_dir', default='test')
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()


with open(args.data_dir + "/bboxes.pickle", 'rb') as f:
    bbox_dict = pickle.load(f)


n = 0


def svhn_generator(batch_size, for_display=False):
    source_data = []

    for key,bboxes in bbox_dict.iteritems():
        for x,y,w,h,label in bboxes:
            source_data.append((key,(x,y,w,h),label))

    current_index = 0

    while True:
        X = []
        y = []
        for i in range(batch_size):
            key, (bx,by,w,h), label = source_data[current_index]
            img = open_image(key)
            img = img.crop((bx,by,bx+w,by+h))
            img = img.resize((32,32), resample=Image.BILINEAR)
            img = img.convert('L')

            if for_display:
                X.append(img)
            else:
                X.append(np.array(img).astype(np.float32) / 255.0)

            y.append(label)
            current_index = (current_index + 1) % len(source_data)

        if for_display:
            yield X,y
        else:
            yield np.array(X).reshape((-1,32,32,1)),to_categorical(y,10)


if args.plot:
    char_width = 32
    char_height = 32
    overview_image = Image.new("L", (char_width * args.char_cols, char_height * args.char_rows), 255)
    overview_draw = ImageDraw.Draw(overview_image)
    generator = svhn_generator(args.char_cols, for_display=True)
    for j in range(0,args.char_rows):
        img_array, y_array = generator.next()
        for i in range(0,args.char_cols):
            overview_image.paste(img_array[i], (char_width*i, char_height*j))
            overview_draw.text((i * char_width, j * char_height + 10), str(y_array[i]))

    overview_image.save("svhn-overview.png")

else:
    print "loading model..."
    model=load_model("test_classifier.hdf5")
    print model.metrics_names

    print "loading test data..."
    generator = svhn_generator(args.n)
    X,y = generator.next()

    print "evaluating model..."
    print model.evaluate(X,y)
