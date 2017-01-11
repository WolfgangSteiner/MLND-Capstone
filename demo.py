import cv2
from PIL import Image, ImageFilter
import numpy as np
from keras.models import load_model
from time import sleep
from detect_text import scan_image
import cProfile
from Drawing import scale_image
from Utils import mkdir, uuid_file_name

pr = cProfile.Profile()

char_size = 32

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


def draw_rect(cv_img, r, color):
    p1 = (int(r.x1),int(r.y1))
    p2 = (int(r.x2),int(r.y2))
    cv2.rectangle(cv_img, p1, p2, color=color, thickness=1)


def draw_answer(cv_img, text, r):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_green = (0,255,0)
    draw_rect(cv_img, r, color_green)
    x = int(r.x1)
    y = int(r.y2)
    cv2.putText(cv_img, text, (x,y), font, fontScale=1, color=color_green, thickness=1)


def draw_detection(cv_img, rect_array):
    for r in rect_array.separate_list:
        draw_rect(cv_img, r, (0,0,128))


def draw_answers(cv_img, result_array):
    for rect,text in result_array:
        draw_answer(cv_img, text, rect)


def draw_probability(cv_img, p):
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, channels = cv_img.shape

    x = 0
    y = height - 10

    color_green = (0,255,0)
    cv2.putText(cv_img, "%.3f" % p, (x,y), font, fontScale=1, color=color_green, thickness=1)


def save_screenshot(cv_img):
    path = "screenshots"
    mkdir(path)
    filename = path + "/" + uuid_file_name("png")
    cv2.imwrite(filename, cv_img)
    print "saving image %s..." % filename

#model=load_model('models/train014-svhn.hdf5')

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 2)
#pr.enable()
is_first = True
while(True):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(gray)
    if is_first:
        print "Image size: %d, %d" % (image.size[0],image.size[1])
        is_first = False
    image = image.filter(ImageFilter.GaussianBlur(radius=1.0))

    for scale in (2.0, 1.0, 0.25):
        result_array, rect_array = scan_image(image, scale, scale)
        draw_detection(frame, rect_array)
        draw_answers(frame, result_array)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        save_screenshot(frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#pr.disable()
#pr.print_stats(sort='time')
