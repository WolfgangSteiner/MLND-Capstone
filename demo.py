import cv2
from PIL import Image, ImageFilter
import numpy as np
from keras.models import load_model
from time import sleep
from detect_text import scan_image
from Drawing import scale_image
from Utils import mkdir, uuid_file_name


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
    for rect,text,_ in result_array:
        draw_answer(cv_img, text, rect)


def draw_segmentation(cv_img, result_array):
    for r,text,seg_array in result_array:
        factor = 32.0 / r.height()
        for s in seg_array:
            x = int(r.x1 + s / factor)
            cv2.line(cv_img, (x, int(r.y1)), (x, int(r.y2)), color=(255,128,128), thickness=2)


def save_screenshot(cv_img):
    path = "screenshots"
    mkdir(path)
    filename = path + "/" + uuid_file_name("png")
    cv2.imwrite(filename, cv_img)
    print "saving image %s..." % filename


cap = cv2.VideoCapture(0)
is_first = True
frame_counter = 0
frame_skip = 8
while(True):
    ret, frame = cap.read()
    frame_counter = (frame_counter + 1) % frame_skip
    if frame_counter % frame_skip:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = Image.fromarray(gray)
    if is_first:
        print "Image size: %d, %d" % (image.size[0],image.size[1])
        is_first = False
        max_factor = 640.0 / image.size[0]
        min_factor = max_factor * 0.1

    result_array, rect_array = scan_image(\
        image, max_factor, min_factor,\
        detector_scaling_factor=0.75, detector_overlap=2.5, detector_threshold=0.75)

    #draw_detection(frame, rect_array)
    #draw_segmentation(frame, result_array)
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
