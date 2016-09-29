import CharacterGenerator
import keras
from keras.layers import Input, Dense, Convolution2D, Flatten, Reshape
from keras.models import Model

# this returns a tensor
inputs = Input(shape=(24,24,1))
x = Convolution2D(64, 3, 3, border_mode="same", activation="relu")(inputs)
x = Convolution2D(64, 3, 3, border_mode="same", activation="relu")(x)
x = Flatten()(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(input=inputs, output=predictions)
model.compile(optimizer='adagrad',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

generator = CharacterGenerator.CharacterGenerator(32)
model.fit_generator(generator, 1000, 40)  # starts training
