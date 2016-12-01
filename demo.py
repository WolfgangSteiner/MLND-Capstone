import cv2
from PIL import Image, ImageFilter
import numpy as np
from keras.models import load_model
from time import sleep
from detect_text import scan_image
import cProfile

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


def draw_rect(cv_img, rect):
    p1 = (int(rect[0]),int(rect[1]))
    p2 = (int(rect[2]),int(rect[3]))
    color_green = (0,255,0)
    cv2.rectangle(cv_img, p1, p2, color=color_green, thickness=1)


def draw_answer(cv_img, text, rect):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_green = (0,255,0)
    draw_rect(cv_img, rect)
    x = int(rect[0])
    y = int(rect[3])
    cv2.putText(cv_img, text, (x,y), font, fontScale=1, color=color_green, thickness=1)


def draw_answers(cv_img, answers):
    for rect,text in answers:
        draw_answer(cv_img, text,rect)


def draw_probability(cv_img, p):
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, channels = cv_img.shape

    x = 0
    y = height - 10

    color_green = (0,255,0)
    cv2.putText(cv_img, "%.3f" % p, (x,y), font, fontScale=1, color=color_green, thickness=1)


#model=load_model('models/train014-svhn.hdf5')

cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FPS, 2)
#pr.enable()

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(gray)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    result_array = scan_image(image, 0.5, 0.125)
    draw_answers(frame, result_array)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#pr.disable()
#pr.print_stats(sort='time')
