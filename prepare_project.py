import os
import urllib
import Utils


Utils.download_and_extract("fonts-master", "master.zip", "https://github.com/google/fonts/archive/master.zip")

import CharacterGenerator
import CharacterSegmentationGenerator
print("Generating character classifier test data...")
options={'min_color_delta':16, 'min_blur':0.5, 'max_blur': 0.5, 'max_rotation':5.0, 'min_noise':4, 'min_size':0.5, 'max_size':1.0, 'max_noise':8, 'full_alphabet':False}
CharacterGenerator.generate_test_data("char_classifier_test.pickle", 4096, options=options)

print("Generating character segmentation+ test data...")
CharacterSegmentationGenerator.generate_test_data("char_segmentation_test.pickle", 4096, options=options)


# from TextImageGenerator import create_test_images
#
# print
# print("Creating test images...")
# create_test_images("TestImagesClean", 2.5, 1)
# create_test_images("TestImagesCleanRotated", 7.5, 1)
# create_test_images("TestImagesNoisy", 2.5, 64)
# create_test_images("TestImagesNoisyRotated", 7.5, 64)
