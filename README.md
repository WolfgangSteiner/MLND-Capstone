# MLND-Capstone: Photo OCR Prototype
![](https://github.com/WolfgangSteiner/MLND-Capstone/blob/master/latex/fig/screenshots/03e8512f-35b8-4446-bc66-e0034f329101.png)

# Instructions
* Execute `prepare_project.py`, which will download and extract fonts from https://github.com/google/fonts and will create
test images for the OCR pipeline.
* Run text detection on the test data with `test_ocr.py`.
* Run `demo.py` with a webcam connected to the computer. Digits will be detected/transcribed in the webcam feed. **Requires GPU!**

# Dependencies
* keras / tensorflow
* OpenCV (for the webcam demo)
* Pillow
* numpy
