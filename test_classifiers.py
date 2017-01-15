import os
import Utils
from plot_data import plot_data
from test_model import test_model

Utils.mkdir("test")
num_examples = 2**13
options={'min_color_delta':16, 'min_blur':0.5, 'max_blur': 1.5, 'max_rotation':7.5, 'min_noise':4, 'min_size':0.5, 'max_size':1.0, 'max_noise':8, 'full_alphabet':False}

if not os.path.exists("test/classifier.pickle"):
    import CharacterGenerator
    print("Generating character classifier test data...")
    CharacterGenerator.generate_test_data("test/classifier.pickle", num_examples, options=options)
    plot_data("test/classifier.pickle")

test_model("models/classifier.hdf5", "test/classifier.pickle")


if not os.path.exists("test/segmentation.pickle"):
    import CharacterSegmentationGenerator
    print("Generating character segmentation test data...")
    CharacterSegmentationGenerator.generate_test_data("test/segmentation.pickle", num_examples, options=options)
    plot_data("test/segmentation.pickle")

test_model("models/segmentation-2.hdf5", "test/segmentation.pickle")


if not os.path.exists("test/detection.pickle"):
    import CharacterDetectionGenerator
    print("Generating character detection test data...")
    CharacterDetectionGenerator.generate_test_data("test/detection.pickle", num_examples, options=options)
    plot_data("test/detection.pickle")

test_model("models/detection.hdf5", "test/detection.pickle")
