import os
import urllib
from zipfile import ZipFile
import Utils

if not os.path.exists("fonts-master"):
    if not os.path.exists("master.zip"):
        print("Downloading fonts from https://github.com/google/fonts ...")
        Utils.download("https://github.com/google/fonts/archive/master.zip", "master.zip")
    else:
        print("Found fonts archive master.zip")

    print("Upacking fonts...")
    zf = ZipFile("master.zip")
    zf.extractall()


from TextImageGenerator import create_test_images

print
print("Creating test images...")
create_test_images("TestImagesClean", 2.5, 1)
create_test_images("TestImagesCleanRotated", 7.5, 1)
create_test_images("TestImagesNoisy", 2.5, 64)
create_test_images("TestImagesNoisyRotated", 7.5, 64)
