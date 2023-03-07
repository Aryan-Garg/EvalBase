#!/usr/bin/python
import os
from pathlib import Path

import cv2 as cv
import numpy as np

import tonemap as tm

# Enable EXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def saveImage(filename, image):
    cv.imwrite(filename, image.astype(np.float32), [cv.IMWRITE_EXR_TYPE, cv.IMWRITE_EXR_TYPE_HALF])

def loadImage(filename, imreadFlags=None):
    return cv.imread(filename, (cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH | cv.IMREAD_UNCHANGED))

# write a function to save exr images as png images
def exr2png(exr_path, png_path):
    exr = tm.tm_display.tonemap(cv.cvtColor(loadImage(exr_path), cv.COLOR_BGR2RGB))
    saveImage(png_path, exr)

# write a loop to convert all exr images in a folder to png images
def exr2png_folder(exr_folder, png_folder):
    for exr_file in os.listdir(exr_folder):
        exr_path = os.path.join(exr_folder, exr_file)
        png_path = os.path.join(png_folder, exr_file[:-4] + ".png")
        exr2png(exr_path, png_path)

# call exr2png_folder to convert all exr images in a folder to png images
exr_folder = "datasets/synthesized_image/"
png_folder = "datasets/synthesized_image_png/"
exr2png_folder(exr_folder, png_folder)
