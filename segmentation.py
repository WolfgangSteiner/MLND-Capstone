from CharacterSequenceGenerator import create_char_sequence
import numpy as np
import pickle, sys, argparse
from keras.models import load_model
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTransform, ImageChops
import cProfile
#pr = cProfile.Profile()
#pr.enable()

image_width = 1024
image_height = 32
num_char_columns = 2
num_char_rows = 32
segmentation_width = 16

print "Loading segmentation classifier..."
segmentation_classifier = load_model("seg006b-word_end.hdf5")
print "Loading character classifier..."
character_classifier = load_model("train049-resize.hdf5")

#segmentation_classifier = load_model("seg001.hdf5")


def prepare_image_for_classification(image):
    w,h = image.size
    image_data = np.array(image).astype('float32')
#    m = np.mean(image_data, axis=(0,1))
#    s = np.std(image_data, axis=(0,1))
#    image_data = (image_data - m) / s
    image_data = image_data.reshape(1,h,w,1)
    return image_data

#    return Image.fromarray(image_data).convert('L')


#def convert_image_for_classification(image):

def check_segmentation(img, pos):
    window_image = img.crop((pos[0],pos[1],pos[0]+segmentation_width,pos[1]+32))
    window_data = prepare_image_for_classification(window_image)
    y = segmentation_classifier.predict(window_data)
    return y[0]


def expand_image_for_segmentation(img):
    (w,h) = img.size
    left_edge = img.crop((0,0,1,image_height))
    right_edge = img.crop((w-1,0,w,image_height))

    expanded_image = Image.new("L", (w + segmentation_width, h), 0)
    (w,h) = expanded_image.size

    for x in range(0,segmentation_width / 2):
        expanded_image.paste(left_edge, (x,0))
        expanded_image.paste(right_edge, (w - x - 1,0))

    expanded_image.paste(img, (segmentation_width/2,0))
    #expanded_image.save("expanded.png")

    return expanded_image


def segment_characters(img, threshold=0.35):
    #img = prepare_image_for_classification(img)
    expanded_image = expand_image_for_segmentation(img)
    (w,h) = expanded_image.size
    x = 0
    is_in_run = False
    x_start = 0
    last_score = 0
    a = 0.99
    b = 1.0 - a
    filtered_score = 0
    seg_array = []
    score_array = []
    while x < w - segmentation_width:
        score = check_segmentation(expanded_image, [x,0])
        filtered_score = score * a + last_score * b
        last_score = filtered_score
        if filtered_score >= threshold and not is_in_run:
            is_in_run = True
            x_start = x
        elif filtered_score < 0.5 * threshold and is_in_run:
            is_in_run = False
            seg_array.append((x_start + x - 1) / 2)

        score_array.append((x,32 * (1.0 - filtered_score)))
        x += 1
    if is_in_run:
        seg_array.append((x_start + x - 2) / 2)

    return seg_array, score_array


def draw_segmentation(img, seg_array):
    draw = ImageDraw.Draw(img)
    for x in seg_array:
        draw.line(((x,0),(x,32)), fill=(0,255,0))


def draw_score(img, score_array):
    draw = ImageDraw.Draw(img)
    for x,y in score_array:
        draw.point((x,y), fill=(255,0,0))


def draw_answer(img, text, predicted_text):
    draw = ImageDraw.Draw(img)
    color = (255,0,0) if text != predicted_text else (0,255,0)
    draw.text((0,20), predicted_text, fill=color)


def classify_character(img, x1, x2):
    char_image = img.crop((x1,0,x2,32)).resize((32,32), resample=Image.BICUBIC)
    char_data = prepare_image_for_classification(char_image)
    ans_vector = character_classifier.predict(char_data)[0]
    ans = ans_vector.argmax()
    probability = ans_vector[ans]
    return str(ans), probability


def classify_characters(img, seg_array, threshold=0.1):
    text = ""
    for i in range(0,len(seg_array)-1):
        char, probability = classify_character(img, seg_array[i], seg_array[i+1])
        if probability > threshold:
            text += char
        else:
            text += 'X'

    return text


def preprocess_image(image):
    width, height = image.size
    min_dim = min(width,height) / 2
    image = image.crop()

    left = (width - min_dim)/2
    top = (height - min_dim)/2
    right = (width + min_dim)/2
    bottom = (height + min_dim)/2
    clip_rect = (left, top, right, bottom)

    image = image.crop(clip_rect).resize((char_size, char_size), Image.LANCZOS)
    image = image.filter(ImageFilter.GaussianBlur(radius=1.0))

    image_data = np.array(image).astype('float32')
    m = np.mean(image_data, axis=(0,1))
    s = np.std(image_data, axis=(0,1))
    image_data = (image_data - m) / s
    image_data = image_data.reshape(1,char_size,char_size,1)
    return image_data, clip_rect



def draw_chars(img, seg_array):
    result = Image.new("RGB", (image_width/2, image_height), 0)
    draw = ImageDraw.Draw(result)
    xi = 0
    char_width = 32
    char_height = 32

    for i in range(0,len(seg_array)-1):
        x1 = seg_array[i]
        x2 = seg_array[i+1]
        char_image = img.crop((x1,0,x2,char_height))
        char_image = char_image.resize((char_width,char_height),resample=Image.BICUBIC)
        result.paste(char_image, (xi, 0))
        draw.line(((xi,0),(xi,32)), fill=(0,255,0))
        xi += char_width

    return result


def test_segmentation(max_num=1024*1024, visualize=False, data_dir="data"):
    options={'min_color_delta':16.0, 'min_blur':0.5, 'max_blur':0.5, 'max_rotation':0.0, 'min_noise':4, 'max_noise':4, 'add_background_lines':False}
    n = 0
    correct_predictions = 0
    file = open(data_dir + '/' + 'labels.pickle', 'rb')
    labels = pickle.load(file)

    if visualize:
        num_examples = min(max_num, len(labels))
        num_char_rows = num_examples / 2 + num_examples % 2
        overview_image = Image.new("RGB", (image_width, 2 * image_height * num_char_rows), (0,0,0))
        overview_draw = ImageDraw.Draw(overview_image)


    for id,text in labels.iteritems():
        img = Image.open(data_dir + '/' + id + ".png")
        #img,text = create_char_sequence(image_width, image_height, options)
        seg_array, score_array = segment_characters(img)
        predicted_text = classify_characters(img, seg_array)

        if visualize:
            x = (n % 2) * image_width / 2
            y = (n / 2) * image_height * 2
            img = img.convert(mode='RGB')
            char_image = draw_chars(img, seg_array)
            draw_score(img, score_array)
            draw_segmentation(img, seg_array)
            draw_answer(img, text, predicted_text)
            overview_image.paste(img, (x, y))
            overview_image.paste(char_image, (x, y + image_height))

        if text == predicted_text:
            correct_predictions += 1

        n+=1
        sys.stdout.write("\r%d/%d  acc: %f" % (correct_predictions, n, float(correct_predictions) / n))
        sys.stdout.flush()

        if n >= max_num:
            break

            #overview_image.paste(Image.fromarray((batch[0][i].reshape(image_height,image_width) * 255).astype('uint8'),mode="L"), (image_width*i, image_height*j))

    if visualize:
        overview_image.save("overview.png")

    print "Accuracy: %d/%d = %f" % (correct_predictions, n, float(correct_predictions) / n)


parser = argparse.ArgumentParser()
parser.add_argument('-n', help="max number of test cases", action="store", dest="n", type=int, default=1024*1024)
parser.add_argument('--directory', help="directory of test cases", action='store', dest='data_dir', default='data')
parser.add_argument("--visualize", help="save visualization of char segmentation and classification", action="store_true", default=False)
args = parser.parse_args()

test_segmentation(max_num=args.n, visualize=args.visualize, data_dir=args.data_dir)


#pr.disable()
#pr.print_stats(sort='time')
