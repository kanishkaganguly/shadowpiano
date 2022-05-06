#!/usr/bin/env python3
import rospy
import rospkg
import numpy as np
from sensor_msgs.msg import Image
import cv2

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

def crop_keys2(img,template,key_template):
    h_t,w_t,_ = template.shape
    h_i,w_i,_ = img.shape
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kpt1, kpt2, good_matches = get_good_matches(img,template)
    matchesMask, H = refine_matches_and_find_homography(kpt1, kpt2, good_matches)
    
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

    return img_cut

def crop_keys(img,template,key_template):
    h_t,w_t,_ = template.shape
    h_i,w_i,_ = img.shape
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kpt1, kpt2, good_matches = get_good_matches(img,template)
    matchesMask, H = refine_matches_and_find_homography(kpt1, kpt2, good_matches)
    
    imgw = cv2.warpPerspective(img, H, (w_t, h_t))
    imgw_gray = cv2.warpPerspective(img_gray, H, (w_t, h_t))
    result = cv2.matchTemplate(imgw_gray, key_template, cv2.TM_SQDIFF_NORMED)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX,-1)
    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    matchLoc = minLoc
    blank = np.zeros(imgw.shape[:2],dtype=np.uint8)
    cv2.rectangle(blank, matchLoc, (matchLoc[0] + key_template.shape[1], matchLoc[1] + key_template.shape[0]), 255, -1)
    
    
    img_cut = cv2.bitwise_and(imgw,imgw,mask = blank)

    return img_cut

def get_ids(key_array):
    param1 = 3
    param2 = 25
    key_start_count = 0
    edge_count = 0
    edge_flag = False
    ids = []
    for i in range(key_array.shape[0]):
        if key_array[i] and not edge_flag:
            edge_count += 1
        if edge_count > param1:    
            edge_flag = True
            edge_count = 0

        if key_array[i] == 0 and edge_flag:
            key_start_count += 1
        if key_start_count > param2:
            ids.append(i-key_start_count)
            edge_flag = False
            key_start_count = 0
    return ids

def get_key_ids(ids):
    key_ids = []
    for i in range(len(ids)-1):
        key_ids.append((ids[i]+ids[i+1])//2)
    return key_ids

def get_keys(img,template,key_template):
    h_t,w_t,_ = template.shape
    h_i,w_i,_ = img.shape
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kpt1, kpt2, good_matches = get_good_matches(img,template)
    matchesMask, H = refine_matches_and_find_homography(kpt1, kpt2, good_matches)
    
    imgw = cv2.warpPerspective(img, H, (w_t, h_t))
    imgw_gray = cv2.warpPerspective(img_gray, H, (w_t, h_t))
    result = cv2.matchTemplate(imgw_gray, key_template, cv2.TM_SQDIFF_NORMED)
    cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX,-1)
    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    matchLoc = minLoc
    xmin = matchLoc[0] - 20
    ymin = matchLoc[1]
    xmax = matchLoc[0] + key_template.shape[1] + 20
    ymax = matchLoc[1] + key_template.shape[0]
    img_just_keys = imgw_gray[ymin:ymax,xmin:xmax]
    grad_x = cv2.Sobel(img_just_keys, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    img_cut1 = cv2.convertScaleAbs(grad_x)
    _,img_cut1 = cv2.threshold(img_cut1,50,255,cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,2))
    img_cut1 = cv2.morphologyEx(img_cut1, cv2.MORPH_CLOSE, kernel)
    
    h,w = img_cut1.shape
    y = h-50
    key_array = img_cut1[y,:]
    ids = get_ids(key_array)
    key_ids = get_key_ids(ids)
    y += ymin
    points = []
    for x in key_ids:
        x += xmin
        points.append([x,y,1])
    
    points = np.linalg.inv(H) @ np.array(points).T 
    points = points.T
    points = points/(points[:,2].reshape(-1,1))
    key_points = points[:,0:2]
    print(key_points)
    return key_points.astype(int)

class piano_observer:
    def __init__(self,template,key_template):
        self.template = template
        self.key_template = key_template
        self.img_front = None
        self.img_left = None
        self.sub_front_cam = rospy.Subscriber("/front_logitech_webcam/image_rect_color",Image,self.callback_front,queue_size=10)
        self.sub_left_cam = rospy.Subscriber("/left_logitech_webcam/image_rect_color",Image,self.callback_left,queue_size=10)

    def callback_front(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height,msg.width,-1)
        self.img_front = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def callback_left(self, msg):
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height,msg.width,-1)
        self.img_left = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def locate_keys(self):
        if type(self.img_front) != type(None):
            key_points = get_keys(self.img_front,self.template,self.key_template)
            return True, key_points
        else:
            return False, None

    def detect_keys(self):
        
        if type(self.img_front) != type(None) and type(self.img_left) != type(None):
            imgf = crop_keys(self.img_front,self.template,self.key_template)
            imgl = crop_keys(self.img_front,self.template,self.key_template)
            img_keys = np.hstack((imgf,imgl))
            return True, img_keys
        else:
            return False, None


def main():
    rospy.init_node("key_locator_node",
                    log_level=rospy.INFO,
                    anonymous=True)
    
    data_path = rospkg.RosPack().get_path('shadowhand') + '/data/'
    template_path = data_path + "piano.png"
    key_template_path = data_path + "piano_keys.png"

    template = cv2.imread(template_path)
    key_template = cv2.imread(key_template_path)
    key_template = cv2.cvtColor(key_template,cv2.COLOR_BGR2GRAY)

    located_keys_status = False
    viewer = piano_observer(template,key_template)
    
    while not located_keys_status:
        located_keys_status, key_points = viewer.locate_keys()
        _,img_keys = viewer.detect_keys()
    
    img = (viewer.img_front).copy()
    for x,y in key_points:
        img = cv2.circle(img, (x,y), 10, (0,0,255), -1)
    cv2.namedWindow('Warp', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Warp',500,500)    
    cv2.imshow("Warp",img)
    cv2.waitKey(0)
    cv2.destroyWindow('Warp')
    

if __name__ == "__main__":
    main()