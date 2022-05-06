import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

def main():
    images_path = "../data"
    types = ('*.jpg', '*.png', '*.jpeg')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(images_path, files)))
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image',1500,500)
    
    for img_path in image_path_list:
        img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,img_thresh1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
        img_thresh2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                            cv2.THRESH_BINARY,11,2)
        img_thresh = np.hstack((img_gray,img_thresh1,img_thresh2))
        img
        cv2.imshow('Image',img_thresh)
        cv2.waitKey(0)  
    cv2.destroyWindow('Image')

if __name__ == "__main__":
    main()