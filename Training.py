from CharacterGenerator import CharacterGenerator
from keras.layers import Input, merge
from keras.layers import Dense, Dropout, Convolution2D, Flatten, Reshape, Activation
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adagrad
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.callbacks import TensorBoard, CSVLogger
from Common import load_svhn
import numpy as np
import random
from keras.layers.core import SpatialDropout2D
from sys import argv

num_classes = 10 #+ 26*2

def create_inception(depth, input_shape):
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

def prepare_svhn(options={}):
    n_val = options.get('n_val', 600)
    n_train = options.get('n_train', 1600)
    num_extra_data = options.get('num_extra_data', None)

    print "Loading data..."
    X_train,y_train = load_svhn("train_32x32.mat")
    X_extra,y_extra = load_svhn("extra_32x32.mat")

    print "Preparing data..."
    num_classes = 10

    n1 = n_val * 2 / 3
    n2 = n_val / 3
    n_val = n1 + n2
    y_val = np.empty((n_val * num_classes,num_classes))
    X_val = np.empty((n_val * num_classes, 32, 32, 1))

    idx_val_train = []
    idx_val_extra = []

    for c in range(0,num_classes):
        print("%d " % c)
        idx1 = np.where(y_train[:,c]==1.0)[0]
        idx2 = np.where(y_extra[:,c]==1.0)[0]
        np.random.shuffle(idx1)
        np.random.shuffle(idx2)
        idx1 = idx1[:n1]
        idx2 = idx2[:n2]
        idxi = n_val * c

        y_val[idxi:idxi+n1] = y_train[idx1]
        y_val[idxi+n1:idxi+n_val] = y_extra[idx2]
        X_val[idxi:idxi+n1] = X_train[idx1]
        X_val[idxi+n1:idxi+n_val] = X_extra[idx2]

        idx_val_train += idx1.tolist()
        idx_val_extra += idx2.tolist()

    X_train = np.delete(X_train, idx_val_train, axis=0)
    y_train = np.delete(y_train, idx_val_train, axis=0)
    X_extra = np.delete(X_extra, idx_val_extra, axis=0)
    y_extra = np.delete(y_extra, idx_val_extra, axis=0)

    if num_extra_data == None:
        X_train = np.append(X_train, X_extra, axis=0)
        y_train = np.append(y_train, y_extra, axis=0)
    elif num_extra_data > 0:
        id_extra = range(0,X_extra.shape[0])
        np.random.shuffle(id_extra)
        id_extra = id_extra[:num_extra_data]
        X_train = np.append(X_train, X_extra[id_extra], axis=0)
        y_train = np.append(y_train, y_extra[id_extra], axis=0)


    print("\n")
    return X_train, y_train, X_val, y_val


class Training(object):
    def __init__(self, batch_size=32, input_shape=(32,32,1), mean=None, std=None):
        self.model = Sequential()
        self.batch_size = batch_size
        self.input_shape=input_shape
        self.mean = mean
        self.std = std
        self.is_first_layer = True
        self.is_first_dense_layer = True
        self.winit = 'glorot_normal'
        self.wreg = 0.01
        self.use_batchnorm = True
        self.output_file_stem = argv[0].split(".")[0]
        self.generator_options = {}
        self.optimizer = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        self.model_checkpoint = ModelCheckpoint(self.output_file_stem + ".hdf5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
        self.tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=False, write_images=False)
        self.csv_logger = CSVLogger(self.output_file_stem + ".log")


    def callbacks(self, options):
        result = []
        patience = options.get('lr_patience', 4)
        factor = options.get('lr_factor', 0.5)
        cooldown = options.get('lr_cooldown', 4)
        min_lr = options.get('lr_min', 0)
        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=patience, verbose=1, mode='auto', epsilon=0.0001, cooldown=cooldown, min_lr=min_lr)
        result.append(reduce_learning_rate)

        for c in (self.tensorboard, self.model_checkpoint, self.csv_logger, self.tensorboard):
            if not c is None:
                result.append(c)
        return result

    def compile(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    def conv(self, depth, filter_size=3):
        if self.is_first_layer:
            conv_layer = Convolution2D(depth, filter_size, filter_size, border_mode='same', W_regularizer=l2(self.wreg), input_shape=self.input_shape)
            self.is_first_layer = False
        else:
            conv_layer = Convolution2D(depth, filter_size, filter_size, border_mode='same', W_regularizer=l2(self.wreg))

        self.model.add(conv_layer)
        if self.use_batchnorm:
            self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

    def inception(self, depth):
        self.is_first_layer = False
        self.model.add(create_inception(depth, self.input_shape))

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

    def avgpool(self):
        self.model.add(AveragePooling2D())


    def dropout(self, p):
        self.model.add(Dropout(p))

    def train_generator(self, options={}):
        self.compile()
        X_val, y_val = CharacterGenerator(2048, self.generator_options).next()
        generator = CharacterGenerator(self.batch_size, self.generator_options)

        self.model.fit_generator(
            generator, 16384, 1000,
            validation_data = (X_val, y_val),
            nb_val_samples = None,
            callbacks = self.callbacks(options),
            max_q_size=16, nb_worker=8, pickle_safe=True)  # starts training

    def train_svhn(self, options={}):
        self.compile()
        X_train, y_train, X_val, y_val = prepare_svhn(options)
        self.model.fit(
            X_train, y_train,
            batch_size=32,
            nb_epoch=500,
            verbose=1,
            callbacks = self.callbacks(options),
            validation_split=0.0,
            validation_data=(X_val,y_val),
            shuffle=True,
            class_weight=None,
            sample_weight=None)

    def train_both(self, **options):
        self.compile()
        X_train_svhn, y_train_svhn, X_val_svhn, y_val_svhn = prepare_svhn(options)
        X_train_gen, y_train_gen = CharacterGenerator(X_train_svhn.shape[0]).next()
        X_val_gen, y_val_gen = CharacterGenerator(X_val_svhn.shape[0]).next()

        X_train = np.append(X_train_svhn, X_train_gen, axis=0)
        y_train = np.append(y_train_svhn, y_train_gen, axis=0)
        X_val = np.append(X_val_svhn, X_val_gen, axis=0)
        y_val = np.append(y_val_svhn, y_val_gen, axis=0)

        self.model.fit(
            X_train, y_train,
            batch_size=32,
            nb_epoch=500,
            verbose=1,
            callbacks = self.callbacks(options),
            validation_split=0.0,
            validation_data=(X_val,y_val),
            shuffle=True,
            class_weight=None,
            sample_weight=None)
