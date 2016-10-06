import numpy as np
from scipy.io import loadmat
from keras.utils.np_utils import to_categorical


def load_svhn(file_name):
    mat = loadmat(file_name)

    y = mat['y']
    y[y==10] = 0
    y = y.reshape(len(y),1)
    y = to_categorical(y, 10)
    n = y.shape[0]
    
    X = mat['X'].astype(np.float32)
    X = X.transpose((3,0,1,2)).reshape(-1,32,32,3)
    X = np.mean(X, axis=3) / 255.0

    for i in range(0,n):
        m = np.mean(X[i,...], axis=(0,1))
        s = np.std(X[i,...], axis=(0,1))
        X[i,...] = (X[i,...] - m) / s 

    X = X.reshape(-1,32,32,1)
                                            
    return X,y
