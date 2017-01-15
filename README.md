# MLND-Capstone: Photo OCR Prototype
In this capstone project I present the prototype of a photo OCR (optical character recognition)
pipeline based on a sliding window algorithm that is able to automatically detect and parse
text (digits) in images. The CNN classifiers used in the pipeline have been trained on images synthesized from
a large collection of fonts. A demo application is included that detects/transcribes digits from a webcam.

![](https://github.com/WolfgangSteiner/MLND-Capstone/blob/master/latex/fig/screenshots/eee8d50a-684a-4f54-8e89-3c4e729530c1.png)


# Instructions
* [Report.pdf](Report.pdf)
* Execute `prepare_project.py`, which will download and extract fonts from https://github.com/google/fonts and will create
a font cache.
* Run `demo.py` with a webcam connected to the computer. Digits will be detected/transcribed in the webcam feed. **Requires GPU!**
* Test the whole OCR pipeline with `test_ocr.py`. This will randomly create images with digit sequences that are then detected/transcribed.
* Test the character segmentation on randomly generated text bounding boxes with `test_segmentation.py`. Will place test images in `TestImagesSegmentation`.
* Test the separate classifiers with `test_classifiers.py`, randomly generated test images are generated. Some examples are collected in `test/classifier.png`, `test/segmentation.png` and `test/detection.png`.
* Test the character classifier on svhn test images: `test_char_classifier_on_svhn_test_data.py`. This will download `test.tar.gz` from http://ufldl.stanford.edu/housenumbers/ (264Mb). Some examples of the test images are collected in `svhn/test.png`.

# Dependencies
* keras / tensorflow
* OpenCV
* Pillow
* numpy
* python-levenshtein
