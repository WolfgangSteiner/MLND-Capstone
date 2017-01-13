import os
import urllib
import Utils


Utils.download_and_extract("fonts-master", "master.zip", "https://github.com/google/fonts/archive/master.zip")


from TextImageGenerator import create_test_images

print
print("Creating test images...")
create_test_images("TestImagesClean", 2.5, 1)
create_test_images("TestImagesCleanRotated", 7.5, 1)
create_test_images("TestImagesNoisy", 2.5, 64)
create_test_images("TestImagesNoisyRotated", 7.5, 64)
