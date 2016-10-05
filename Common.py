import numpy as np
from scipy.io import loadmat
from keras.utils.np_utils import to_categorical


def load_svhn(file_name):
    mat = loadmat(file_name)
    X = mat['X'].astype(np.float32)
    y = mat['y']
    y = y.reshape(len(y),1)
    y[y==10] = 0
    y = to_categorical(y, 10)
    X = np.mean(X, axis=2) / 255.0
    X = X.transpose((2,0,1)).reshape(-1,32,32,1)
    return X,y
