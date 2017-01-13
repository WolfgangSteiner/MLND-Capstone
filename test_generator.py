import numpy as np
from keras.models import load_model
import CharacterGenerator
from sys import argv

char_size = 32

print "Loading model..."
model=load_model(argv[1])
print "Generating test data..."
X,y = CharacterGenerator.CharacterGenerator(8192, {'show_progress':True}).next()
print "Evaluating model..."
print model.evaluate(X,y)
