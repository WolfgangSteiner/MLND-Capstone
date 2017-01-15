from TextImageGenerator import create_test_images
from detect_text import test_ocr
import argparse
import os
import Utils

parser = argparse.ArgumentParser()
parser.add_argument('--purge', action="store_true")
args = parser.parse_args()

def create_test_set(name, max_rotation, max_blur):
    if not os.path.exists(name):
        print("Creating test set %s..." % name)
        create_test_images(name, max_rotation, 64, max_blur=max_blur, n=512)
        print("")

test_sets = [\
    ("TestImages", 2.5, 1.5, 0.75, 4.0, 0.625),\
    ("TestImagesBlur", 2.5, 4.0, 0.75, 4.0, 0.625),\
    ("TestImagesRotate", 7.5, 1.5, 0.75, 4.0, 0.625),\
    ("TestImagesBlurRotate", 7.5, 4.0, 0.75, 4.0, 0.625)]

if args.purge:
    for name,_,_ in test_sets:
        Utils.rmdir(name)

for name, max_rotation, max_blur,_,_,_ in test_sets:
    create_test_set(name, max_rotation, max_blur)

results = []

for name,_,_,detector_scaling_factor,detector_overlap,detector_threshold in test_sets:
    results.append(test_ocr(name, detector_scaling_factor=detector_scaling_factor, detector_overlap=detector_overlap, detector_threshold=detector_threshold))


print ("%-25s %+5s %+11s %+10s %+10s %+10s %+10s" % ("Test Set", "N", "Accuracy", "Time", "Factor", "Overlap", "Threshold"))
print ("="*70)
for r in results:
    print ("%-25s %5.0d %10.2f%% %10.0f %f %f %f" % r)
