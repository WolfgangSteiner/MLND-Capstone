import Utils
import os
import sys
from prepare_svhn_digit_data import prepare_svhn_digit_data
from test_model import test_model
from plot_data import plot_data

data_file = "svhn/test.pickle"
if not os.path.exists(data_file):
    Utils.mkdir("svhn")
    print("")
    print("")
    print("This script will test the character classifier on test data from the street view house numbers.")
    print("Downloading and convertig the data might take a while.")
    print("If you already have the file 'http://ufldl.stanford.edu/housenumbers/test.tar.gz', place it in this directory.")

    if os.path.exists("svhn/test") or Utils.query_yes_no("Continue?"):
        Utils.download_and_extract("svhn/test", "svhn/test.tar.gz", "http://ufldl.stanford.edu/housenumbers/test.tar.gz")
        prepare_svhn_digit_data("svhn/test")
        print("Plotting some examples to char_classifier_test_svhn.png ...")
        plot_data(data_file)
    else:
        sys.exit(0)


test_model("models/classifier.hdf5", data_file)
