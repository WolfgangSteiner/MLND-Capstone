from PIL import Image
import numpy as np
from keras.models import load_model
from Common import load_svhn

char_size = 32

def preprocess_image(image):
    width, height = image.size
    min_dim = min(width,height) / 8
    image = image.crop()

    left = (width - min_dim)/2
    top = (height - min_dim)/2
    right = (width + min_dim)/2
    bottom = (height + min_dim)/2
    clip_rect = (left, top, right, bottom)

    image = image.crop(clip_rect).resize((char_size, char_size), Image.LANCZOS)
    image_data = np.array(image).reshape(1,char_size,char_size,1).astype('float32') / 255.0
    return image_data, clip_rect


def draw_clip_rect(cv_img, clip_rect):
    p1 = (clip_rect[0],clip_rect[1])
    p2 = (clip_rect[2],clip_rect[3])
    color_green = (0,255,0)
    cv2.rectangle(cv_img, p1, p2, color=color_green, thickness = 2)

def draw_answer(cv_img, clip_rect, answer):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = (clip_rect[0] + clip_rect[2]) / 2
    y = (clip_rect[1] + clip_rect[3]) / 2
    color_green = (0,255,0)
    cv2.putText(cv_img, str(answer), (x,y), font, fontScale=4, color=color_green, thickness=2)

def draw_probability(cv_img, p):
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width, channels = cv_img.shape

    x = 0
    y = height - 10

    color_green = (0,255,0)
    cv2.putText(cv_img, "%.3f" % p, (x,y), font, fontScale=1, color=color_green, thickness=1)

model=load_model('checkpoint.hdf5')

X,y = load_svhn('test_32x32.mat')
print model.evaluate(X,y)

# cap = cv2.VideoCapture(0)
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     image = Image.fromarray(gray)
#
#     image_data, clip_rect = preprocess_image(image)
#
#     ans_vector = model.predict(image_data)[0]
#     ans = ans_vector.argmax()
#     probability = ans_vector[ans]
#
#     draw_clip_rect(frame, clip_rect)
#
#     if (probability > 0.5):
#         draw_answer(frame, clip_rect, ans)
#
#     draw_probability(frame, probability)
#
#     # Display the resulting frame
#     cv2.imshow('frame', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
