import CharacterGenerator
from keras.layers import Input, merge
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Reshape, Activation
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

num_classes = 26*2 + 10

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

model = Sequential()
model.add(inception(32, (24,12,1)))
model.add(Dropout(0.5))
model.add(MaxPooling2D((3, 3), strides=(2,2), border_mode='same'))
model.add(inception(64, (12,6,4*32)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes, init='uniform'))
model.add(Activation('softmax'))

model.compile(
    optimizer='adagrad',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

batch_size = 32
generator = CharacterGenerator.CharacterGenerator(batch_size)

model_checkpoint = ModelCheckpoint("checkpoint.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

model.fit_generator(
    generator, batch_size * 512, 1000,
    validation_data = generator,
    nb_val_samples = batch_size * 32,
    callbacks = [model_checkpoint, reduce_learning_rate],
    max_q_size=16, nb_worker=8, pickle_safe=True)  # starts training
