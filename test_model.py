import argparse
import pickle
from keras.models import load_model


def test_model(model_file_name, data_file_name, n=None):
    print("loading model %s ..." % model_file_name)
    model=load_model(model_file_name)

    print("loading test data %s ..." % data_file_name)
    with open(data_file_name) as f:
        X = pickle.load(f)
        y = pickle.load(f)

    n = X.shape[0] if n is None else n
    print("evaluating model...")
    loss, accuracy = model.evaluate(X[0:n],y[0:n])
    print("")
    print("Test accuracy: %.2f%%" % (accuracy * 100.0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('test_data')
    parser.add_argument('-n', action="store", dest="n", type=int, default=None)
    args = parser.parse_args()
    test_model(args.model, args.test_data, args.n)
