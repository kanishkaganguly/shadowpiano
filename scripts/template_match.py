import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

def main():
    images_path = "../data"
    types = ('*.jpg', '*.jpeg')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(images_path, files)))
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image',1500,500)
    template = cv2.imread("../data/piano_keys.png")
    template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    for img_path in image_path_list:
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX,-1)
        _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
        matchLoc = minLoc
        cv2.rectangle(img, matchLoc, (matchLoc[0] + template.shape[1], matchLoc[1] + template.shape[0]), (0,0,255), 5, 8, 0 )
        cv2.imshow('Image',img)
        cv2.waitKey(0)  
    cv2.destroyWindow('Image')

if __name__ == "__main__":
    main()