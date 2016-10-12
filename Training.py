import CharacterGenerator
from keras.layers import Input, merge
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Reshape, Activation
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adagrad
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.callbacks import TensorBoard
from Common import load_svhn
import numpy as np
import random
from keras.layers.core import SpatialDropout2D
from sys import argv

num_classes = 10 #+ 26*2

def inception(depth, input_shape):
    input = Input(shape=input_shape)
    tower1 = Convolution2D(depth, 1, 1, border_mode='same', W_regularizer=l2(0.01), init='glorot_normal')(input)
    tower1 = Activation('relu')(tower1)
    tower1 = Convolution2D(depth, 3, 3, border_mode='same', W_regularizer=l2(0.01), init='glorot_normal')(tower1)
    tower1 = Activation('relu')(tower1)

    tower2 = Convolution2D(depth, 1, 1, border_mode='same', W_regularizer=l2(0.01), init='glorot_normal')(input)
    tower2 = Activation('relu')(tower2)
    tower2 = Convolution2D(depth, 5, 5, border_mode='same', W_regularizer=l2(0.01), init='glorot_normal')(tower2)
    tower2 = Activation('relu')(tower2)

    tower3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    tower3 = Convolution2D(depth, 1, 1, border_mode='same', W_regularizer=l2(0.01), init='glorot_normal')(tower3)
    tower3 = Activation('relu')(tower3)

    tower4 = Convolution2D(depth, 1, 1, border_mode='same', W_regularizer=l2(0.01), init='glorot_normal')(input)
    tower4 = Activation('relu')(tower4)
    output = merge([tower1, tower2, tower3, tower4], mode='concat')
    return Model(input, output)


def convnet(d1,d2,d3, input_shape):
    model = Sequential()

    model.add(Convolution2D(d1, 3, 3, border_mode='same', init='glorot_normal', W_regularizer=l2(0.01), input_shape=(32,32,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(SpatialDropout2D(0.5))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(1,1)))

    model.add(Convolution2D(d2, 3, 3, border_mode='same', init='glorot_normal', W_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    #model.add(SpatialDropout2D(0.5))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    model.add(Convolution2D(d3, 3, 3, border_mode='same', init='glorot_normal', W_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))

    return model


def are_elements_unique(a):
    u, i = np.unique(a, return_inverse=True)
    return len(u[np.bincount(i) > 1]) == 0

def prepare_svhn():
    print "Loading data..."
    X_train,y_train = load_svhn("train_32x32.mat")
    X_extra,y_extra = load_svhn("extra_32x32.mat")

    print "Preparing data..."
    num_classes = 10
    n1 = 400
    n2 = 200
    n_val = n1+n2
    y_val = np.empty((n_val * num_classes,num_classes))
    X_val = np.empty((n_val * num_classes, 32, 32, 1))

    idx_val_train = []
    idx_val_extra = []

    for c in range(0,num_classes):
        idx1 = np.where(y_train[:,c]==1.0)[0]
        idx2 = np.where(y_extra[:,c]==1.0)[0]
        np.random.shuffle(idx1)
        np.random.shuffle(idx2)
        idx1 = idx1[:n1]
        idx2 = idx2[:n2]

        idxi = n_val * c

        y_val[c*n_val:c*n_val+n1] = y_train[idx1]
        y_val[c*n_val+n1:(c+1)*n_val] = y_extra[idx2]
        X_val[c*n_val:c*n_val+n1] = X_train[idx1]
        X_val[c*n_val+n1:(c+1)*n_val] = X_extra[idx2]

        idx_val_train += idx1.tolist()
        idx_val_extra += idx2.tolist()

        X_train = np.delete(X_train, idx_val_train, axis=0)
        y_train = np.delete(y_train, idx_val_train, axis=0)
        X_extra = np.delete(X_extra, idx_val_extra, axis=0)
        y_extra = np.delete(y_extra, idx_val_extra, axis=0)

        X_train = np.append(X_train, X_extra, axis=0)
        y_train = np.append(y_train, y_extra, axis=0)

        return X_train, y_train, X_val, y_val


class Training(object):
    def __init__(self, batch_size=32, input_shape=(32,32,1)):
        self.model = Sequential()
        self.batch_size = batch_size
        self.input_shape=input_shape
        self.is_first_layer = True
        self.is_first_dense_layer = True
        self.winit = 'glorot_normal'
        self.wreg = 0.01
        self.use_batchnorm = True
        self.output_file = argv[0].split(".")[0] + ".hdf5"
        print("output_file: " + self.output_file)
        self.generator = CharacterGenerator.CharacterGenerator(batch_size)
        self.optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        self.model_checkpoint = ModelCheckpoint(self.output_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        self.reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', epsilon=0.0001, cooldown=4, min_lr=0)
        self.tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=False, write_images=False)

    def compile(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def conv(self, depth, filter_size=3):
        conv_layer = Convolution2D(depth, filter_size, filter_size, W_regularizer=l2(self.wreg))

        if self.is_first_layer:
            conv_layer = Convolution2D(depth, filter_size, filter_size, W_regularizer=l2(self.wreg), input_shape=self.input_shape)
            self.is_first_layer = False
        else:
            conv_layer = Convolution2D(depth, filter_size, filter_size, W_regularizer=l2(self.wreg))

        self.model.add(conv_layer)
        if self.use_batchnorm:
            self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

    def dense(self, output_size):
        if self.is_first_layer:
            input_shape = self.input_shape
            self.is_first_layer = False
            self.is_first_dense_layer = False

        if self.is_first_dense_layer:
            self.model.add(Flatten())
            self.is_first_dense_layer = False

        self.model.add(Dense(256, init=self.winit, W_regularizer=l2(self.wreg)))

        if self.use_batchnorm:
            self.model.add(BatchNormalization())

        self.model.add(Activation('relu'))


    def classifier(self, num_classes=10):
        self.model.add(Dense(num_classes, init=self.winit))
        self.model.add(Activation('softmax'))


    def maxpool(self):
        self.model.add(MaxPooling2D())


    def dropout(self, p):
        self.model.add(Dropout(p))

    def train_generator(self):
        self.compile()
        self.model.fit_generator(
            self.generator, 16384 * 8, 1000,
            validation_data = CharacterGenerator.CharacterGenerator(4096).next(),
            nb_val_samples = self.batch_size * 32,
            callbacks = [self.model_checkpoint, self.reduce_learning_rate, self.tensorboard],
            max_q_size=16, nb_worker=8, pickle_safe=True)  # starts training

    def train_svhn(self):
        self.compile()
        X_train, y_train, X_val, y_val = prepare_svhn()
        self.model.fit(
            X_train, y_train,
            batch_size=32,
            nb_epoch=500,
            verbose=1,
            callbacks = [self.model_checkpoint, self.reduce_learning_rate, self.tensorboard],
            validation_split=0.0,
            validation_data=(X_val,y_val),
            shuffle=True,
            class_weight=None,
            sample_weight=None)
