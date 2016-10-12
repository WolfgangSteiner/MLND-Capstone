import numpy as np
from keras.models import load_model
from Common import load_svhn
from sys import argv

print "loading model..."
model=load_model(argv[1])
print model.metrics_names
print "loading test data..."
X,y = load_svhn('test_32x32.mat')
print "evaluating model..."
print model.evaluate(X,y)
