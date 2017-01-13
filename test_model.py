import argparse
import pickle
from keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('test_data')
parser.add_argument('-n', action="store", dest="n", type=int, default=None)
args = parser.parse_args()

print "loading model..."
model=load_model(args.model)
print model.metrics_names

print "loading test data..."
with open(args.test_data) as f:
    X = pickle.load(f)
    y = pickle.load(f)

n = X.shape[0] if args.n is None else args.n
print "evaluating model..."
#print model.evaluate(X[0:n],y[0:n])
print model.evaluate(X,y)
