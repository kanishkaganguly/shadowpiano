import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

def get_good_matches(img1,img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # sift
    sift = cv2.xfeatures2d.SIFT_create()

    kpt1, des1 = sift.detectAndCompute(img1,None)
    kpt2, des2 = sift.detectAndCompute(img2,None)
    
    index_params = {'algorithm':1, 'trees':5}
    search_params = {'checks':50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    
    good_matches = []
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good_matches.append(m)
    
    return kpt1, kpt2, good_matches

def refine_matches_and_find_homography(kpt1, kpt2, good_matches):
    pts1 = np.float32([ kpt1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts2 = np.float32([ kpt2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    return matchesMask, H

def draw_matches(img1,img2,kpt1,kpt2,matches,matchesMask,savepath):
    draw_params = {"matchesMask":matchesMask, "flags":2}
    img_matches = cv2.drawMatches(img1,kpt1,img2,kpt2,matches,None,**draw_params)
    # cv2.imwrite(savepath,img_matches)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image',1500,500)
    cv2.imshow('Image',img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    images_path = "../data"
    types = ('*.jpg', '*.jpeg')
    image_path_list= []
    for files in types:
        image_path_list.extend(glob(os.path.join(images_path, files)))
    cv2.namedWindow('Warp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Warp',1000,500)
    template = cv2.imread("../data/piano.png")
    # template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    # _,template = cv2.threshold(template,150,255,cv2.THRESH_BINARY)
    h_t,w_t,_ = template.shape
    key_template = cv2.imread("../data/piano_keys.png")
    key_template = cv2.cvtColor(key_template,cv2.COLOR_BGR2GRAY)
    
    for img_path in image_path_list:
        img = cv2.imread(img_path)
        h_i,w_i,_ = img.shape
        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kpt1, kpt2, good_matches = get_good_matches(img,template)
        matchesMask, H = refine_matches_and_find_homography(kpt1, kpt2, good_matches)
        # draw_matches(img,template,kpt1,kpt2,good_matches,matchesMask,None)
        imgw = cv2.warpPerspective(img, H, (w_t, h_t))
        imgw_gray = cv2.warpPerspective(img_gray, H, (w_t, h_t))
        result = cv2.matchTemplate(imgw_gray, key_template, cv2.TM_SQDIFF_NORMED)
        cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX,-1)
        _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
        matchLoc = minLoc
        blank = np.zeros(imgw.shape[:2],dtype=np.uint8)
        cv2.rectangle(blank, matchLoc, (matchLoc[0] + key_template.shape[1], matchLoc[1] + key_template.shape[0]), 255, -1)
        img_mask = cv2.warpPerspective(blank, np.linalg.inv(H), (w_i, h_i))
        
        img_cut = cv2.bitwise_and(img,img,mask = img_mask)
        comp = np.hstack((img,img_cut))
        cv2.rectangle(imgw, matchLoc, (matchLoc[0] + key_template.shape[1], matchLoc[1] + key_template.shape[0]), (0,0,255), 5, 8, 0 )
        savepath = "../outputs/" + img_path.split("/")[-1]
        cv2.imwrite(savepath,comp)
        cv2.imshow('Warp',comp)
        cv2.waitKey(0)  

    cv2.destroyWindow('Warp')

if __name__ == "__main__":
    main()