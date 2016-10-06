import CharacterGenerator
from keras.layers import Input, merge
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Reshape, Activation
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adagrad
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.constraints import maxnorm
from keras.regularizers import l2, activity_l2
from keras.callbacks import TensorBoard
from Common import load_svhn
import numpy as np
import random


num_classes = 10 #+ 26*2

def inception(depth, input_shape):
    input = Input(shape=input_shape)
    tower1 = Convolution2D(depth, 1, 1, border_mode='same', activation='relu')(input)
    tower1 = Convolution2D(depth, 3, 3, border_mode='same', activation='relu')(tower1)
    tower2 = Convolution2D(depth, 1, 1, border_mode='same', activation='relu')(input)
    tower2 = Convolution2D(depth, 5, 5, border_mode='same', activation='relu')(tower2)
    tower3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    tower3 = Convolution2D(depth, 1, 1, border_mode='same', activation='relu')(tower3)
    tower4 = Convolution2D(depth, 1, 1, border_mode='same', activation='relu')(input)
    output = merge([tower1, tower2, tower3, tower4], mode='concat')
    return Model(input, output)


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


model = Sequential()

model.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu', init='glorot_normal', input_shape=(32,32,1)))
model.add(MaxPooling2D())

model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu', init='glorot_normal'))
model.add(MaxPooling2D())

model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', init='glorot_normal'))
model.add(MaxPooling2D())

#model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', init='glorot_normal'))
#model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu', init='glorot_normal'))
#model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', init='glorot_normal'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, init='glorot_normal'))
model.add(Activation('softmax'))

adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)

model.compile(
    optimizer=adagrad,
    loss='categorical_crossentropy',
    metrics=['accuracy'])


batch_size = 32


model_checkpoint = ModelCheckpoint("checkpoint.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, mode='auto', epsilon=0.0001, cooldown=4, min_lr=0)

tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=False, write_images=False)

generator = CharacterGenerator.CharacterGenerator(batch_size)
model.fit_generator(
    generator, 16384, 1000,
    validation_data = generator,
    nb_val_samples = batch_size * 32,
    callbacks = [model_checkpoint ,reduce_learning_rate, tensorboard],
    max_q_size=16, nb_worker=8, pickle_safe=True)  # starts training


#model.fit(
#    X_train, y_train,
#    batch_size=32,
#    nb_epoch=500,
#    verbose=1,
#    callbacks = [model_checkpoint ,reduce_learning_rate, tensorboard],
#    validation_split=0.0,
#    validation_data=(X_val,y_val),
#    shuffle=True,
#    class_weight=None,
#    sample_weight=None)
