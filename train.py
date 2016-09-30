import CharacterGenerator
from keras.layers import Input, merge
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten, Reshape, Activation
from keras.models import Model, Sequential
from keras.optimizers import SGD

input = Input(shape=(24,24,1))
tower_1 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input)
tower_1 = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(tower_1)
tower_2 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input)
tower_2 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(tower_2)
tower_3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
tower_3 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(tower_3)
tower_4 = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(input)
output = merge([tower_1, tower_2, tower_3, tower_4], mode='concat', concat_axis=1)
inception1 = Model(input, output)

model = Sequential()
model.add(inception1)
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(10, init='uniform'))
model.add(Activation('softmax'))

model.compile(
    optimizer='adagrad',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

batch_size = 32
generator = CharacterGenerator.CharacterGenerator(batch_size)

model.fit_generator(
    generator, batch_size * 512, 1000,
    validation_data = generator,
    nb_val_samples = batch_size * 32,
    max_q_size=16, nb_worker=8, pickle_safe=True)  # starts training



