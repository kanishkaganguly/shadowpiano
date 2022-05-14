#!/usr/bin/env python
import rospy
import rospkg
import numpy as np
import tf
import geometry_msgs.msg
from sensor_msgs.msg import Image, CameraInfo
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
    xmin = matchLoc[0] + 20
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
    y = h-40
    key_array = img_cut1[y,:]
    ids = get_ids(key_array)
    key_ids = get_key_ids(ids)
    y += ymin
    points = []
    for x in key_ids:
        x += xmin
        points.append([x,y,1])
    
    points = np.matmul(np.linalg.inv(H), np.array(points).T) 
    points = points.T
    points = points/(points[:,2].reshape(-1,1))
    key_points = points[:,0:2]
    print(key_points)
    return key_points.astype(int)

def LinearTriangulation(K1, K2, A1, A2, x1, x2):
    # [R1,C1] is from world to cam1
    # [R2,C2] is from world to cam2
    sz = x1.shape[0]
    P1 = np.matmul(K1,A1[0:3,:])
    P2 = np.matmul(K2,A2[0:3,:])

    X = np.zeros((sz, 3))

    for i in range(sz):
        A = np.zeros((4,4))
        A[0,:] = x1[i,0]*P1[2,:] - P1[0,:]
        A[1,:] = x1[i,1]*P1[2,:] - P1[1,:]
        A[2,:] = x2[i,0]*P2[2,:] - P2[0,:]
        A[3,:] = x2[i,1]*P2[2,:] - P2[1,:]
        _, _, Vt = np.linalg.svd(A)
        x = Vt[3,:]
        x = x.reshape(4,)
        x /= x[3]
        X[i, :] = x[0:3].reshape(1,3)

    return X

class piano_observer:
    def __init__(self,template_left,template_front,key_template_left,key_template_front):
        self.template_left = template_left
        self.key_template_left = key_template_left
        self.template_front = template_front
        self.key_template_front = key_template_front
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
        if type(self.img_front) != type(None) and type(self.img_left) != type(None):
            key_points_front = get_keys(self.img_front,self.template_front,self.key_template_front)
            key_points_left = get_keys(self.img_left,self.template_left,self.key_template_left)
            return True, key_points_front, key_points_left
        else:
            return False, None, None

    def detect_keys(self):
        
        if type(self.img_front) != type(None) and type(self.img_left) != type(None):
            print("here")
            imgf = crop_keys(self.img_front,self.template_front,self.key_template_front)
            imgl = crop_keys(self.img_left,self.template_left,self.key_template_left)
            hf,wf,_ = imgf.shape
            hl,wl,_ = imgl.shape
            h = max(hl,hf)
            img_keys = np.zeros((h,wf+wl,3),dtype=np.uint8)
            img_keys[0:hf,0:wf,:] = imgf
            img_keys[0:hl,wf:,:] = imgl
            return True, img_keys
        else:
            return False, None


def main():
    rospy.init_node("key_locator_node",
                    log_level=rospy.INFO,
                    anonymous=True)
    
    data_path = rospkg.RosPack().get_path('shadowpiano') + '/data/'
    template_path_left = data_path + "piano_left.png"
    key_template_path_left = data_path + "piano_keys_left.png"
    template_path_front = data_path + "piano_front.png"
    key_template_path_front = data_path + "piano_keys_front.png"


    template_left = cv2.imread(template_path_left)
    key_template_left = cv2.imread(key_template_path_left)
    key_template_left = cv2.cvtColor(key_template_left,cv2.COLOR_BGR2GRAY)
    template_front = cv2.imread(template_path_front)
    key_template_front = cv2.imread(key_template_path_front)
    key_template_front = cv2.cvtColor(key_template_front,cv2.COLOR_BGR2GRAY)

    located_keys_status = False
    viewer = piano_observer(template_left,template_front,key_template_left,key_template_front)
    tf_listener = tf.TransformListener()
    br = tf.TransformBroadcaster()

    while not located_keys_status:
        located_keys_status, key_points_front, key_points_left = viewer.locate_keys()
        # located_keys_status,img_keys = viewer.detect_keys()

    camera_front_info_msg = rospy.wait_for_message("/front_logitech_webcam/webcam/camera_info", CameraInfo)
    camera_left_info_msg = rospy.wait_for_message("/left_logitech_webcam/webcam/camera_info", CameraInfo)
    K_f = np.array(camera_front_info_msg.K).reshape((3,3))
    K_l = np.array(camera_left_info_msg.K).reshape((3,3))
    K_f = K_f.astype(np.float64)
    K_l = K_l.astype(np.float64)
    K_f_inv = np.linalg.inv(K_f)
    K_l_inv = np.linalg.inv(K_l)

    # # Homogenous Image coordinates
    xf = np.hstack((key_points_front,np.ones((key_points_front.shape[0],1))))
    xl = np.hstack((key_points_left,np.ones((key_points_left.shape[0],1))))
    xf = xf.astype(np.float64)
    xl = xl.astype(np.float64)
    
    (trans_f,rot_f) = tf_listener.lookupTransform('/front_cam', '/world', rospy.Time(0))
    (trans_l,rot_l) = tf_listener.lookupTransform('/left_cam', '/world', rospy.Time(0))
    A_f = tf_listener.fromTranslationRotation(trans_f, rot_f)
    A_l = tf_listener.fromTranslationRotation(trans_l, rot_l)

    X = LinearTriangulation(K_f, K_l, A_f, A_l, xf, xl)

    (trans_ra,rot_ra) = tf_listener.lookupTransform('/world', '/ra_tool0', rospy.Time(0))
    (trans_rh,rot_rh) = tf_listener.lookupTransform('/world', '/rh_fftip', rospy.Time(0))
    x_ra_to_rh = np.array([trans_ra[0]-trans_rh[0],trans_ra[1]-trans_rh[1],trans_ra[2]-trans_rh[2]]).reshape(1,3)
    X[:,2] += 0.01
    X = X + x_ra_to_rh
    rate = rospy.Rate(1000.0)
    while not rospy.is_shutdown():

        br.sendTransform((X[4,0], X[4,1], X[4,2]),rot_ra,rospy.Time.now(),"fifth_key","world")
        br.sendTransform((X[6,0], X[6,1], X[6,2]),rot_ra,rospy.Time.now(),"seventh_key","world")
        br.sendTransform((X[8,0], X[8,1], X[8,2]),rot_ra,rospy.Time.now(),"ninth_key","world")
        rate.sleep()


if __name__ == "__main__":
    main()