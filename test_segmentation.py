from CharacterSequenceGenerator import create_segmentation_examples
from segmentation import test_segmentation
import argparse
import os
import Utils

parser = argparse.ArgumentParser()
parser.add_argument('--purge', action="store_true")
args = parser.parse_args()
data_dir = "TestImagesSegmentation"

if args.purge:
    Utils.rmdir(data_dir)

if not os.path.exists(data_dir):
    print("Creating test set %s..." % data_dir)
    create_segmentation_examples(data_dir, n=512)
    print("")

test_segmentation(data_dir=data_dir)
