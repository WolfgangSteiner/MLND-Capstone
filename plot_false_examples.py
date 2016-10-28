import numpy as np
from keras.models import load_model
from Common import load_svhn
from scipy.io import loadmat
from PIL import Image, ImageDraw
from sys import argv

def load_svhn_unnormalized(file_name):
    mat = loadmat(file_name)
    X = mat['X'].astype(np.float32)
    n = X.shape[0]
    X = X.transpose((3,0,1,2)).reshape(-1,32,32,3)
    X = np.mean(X, axis=3)
    return X

print "loading model..."
model=load_model(argv[1])
print model.metrics_names
print "loading test data..."
X,y = load_svhn('test_32x32.mat')
print "evaluating model..."

y_predict = model.predict(X)
y = np.argmax(y, axis=1)
y_predict = np.argmax(y_predict, axis=1)

false_array = y_predict != y
false_idx = np.where(false_array == True)[0]

pos_x = 0
pox_y = 0
count = 0
X = load_svhn_unnormalized('test_32x32.mat')

char_height = 32
char_width = 32
num_char_columns = 16
num_char_rows = false_idx.shape[0] / num_char_columns + 1

overview_image = Image.new("L", (char_width * num_char_columns, char_height * num_char_rows), 255)
draw = ImageDraw.Draw(overview_image)

for idx in false_idx:
    img = Image.fromarray(X[idx].reshape(32,32)).convert('L')
    label_test = y[idx]
    label_predict = y_predict[idx]
    pos_x = count % num_char_columns
    pos_y = count / num_char_columns
    overview_image.paste(img, (char_width*pos_x, char_height*pos_y))
    draw.text((pos_x * char_width, pos_y * char_height + 20), str(label_test), fill=0 )
    draw.text((pos_x * char_width + 1, pos_y * char_height + 20), str(label_test), fill=255)
    draw.text((pos_x * char_width + 10, pos_y * char_height + 20), str(label_predict), fill=0)
    draw.text((pos_x * char_width + 11, pos_y * char_height + 20), str(label_predict), fill=255)
    count += 1

overview_image.save("false_classified.png")
