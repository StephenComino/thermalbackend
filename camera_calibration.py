#Calibration
from depthcam import DepthCamera
from test import ThermalCamera
import numpy as np
import cv2
# Take Photos on both cameras.
# 1. Depth Camera, Take photos and save them in a location
# 2. 

if __name__=="__main__":
    # Get the Depth Camera
    # Take an image with the camera. RGB style
    dp = DepthCamera()
    dp.captureAndSaveImage()
    
    # Load the Thermal Camera
    # Take an image with the Thermal Camera
    #tc = ThermalCamera()
    #tc.takeThermalImage()
    
    # Rectification
    # Get Matrices
    # A1
    #617.024632 0.000000 81.000000 
    #0.000000 617.024632 61.000000 
    #0.000000 0.000000 1.000000
    # A2
    #245.742320 0.000000 80.500000 
    #0.000000 245.742320 60.500000 
    #0.000000 0.000000 1.000000  
    A1 = np.matrix([[617.024632, 0.000000, 81.000000],
                  [0.000000, 617.024632, 61.000000],
                  [0.000000, 0.000000, 1.000000]])
    A2 = np.matrix([[245.742320, 0.000000, 80.500000],
                  [0.000000, 245.742320, 60.500000],
                  [0.000000, 0.000000,1.000000]])
    #R1
    #0.030802 0.999525 0.000280 
    #0.639697 -0.019929 0.768369 
    #0.768010 -0.023488 -0.640007 
    R1 = np.matrix([[0.030802, 0.999525, 0.000280],
                  [0.639697, -0.019929, 0.768369],
                  [0.768010, -0.023488, -0.640007]])
    #R2
    #-0.097847 0.974322 -0.202788 
    #0.626224 0.218646 0.748356 
    #0.773479 -0.053767 -0.631538
    R2 = np.matrix([[-0.097847, 0.974322, -0.202788],
                  [0.626224, 0.218646, 0.748356],
                  [00.773479, -0.053767, -0.631538]])
    points1 = np.array([[87,33],[88,58],[125,33], [125,61], [126, 82], [82,64], [127,25]])
    points2 = np.array([[62,22],[57,47],[99,28], [93,54], [86,87],[44,67], [93,13]])
    fund, mask = cv2.findFundamentalMat(points1, points2)
    retval, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(points1), np.float32(points2), fund, (160,120))
    print(retval)
    # Load both images
    # warpPerspective()
    
    img_rgb = cv2.imread('/home/pi/Desktop/r2.png')
    img_thermal = cv2.imread('/home/pi/Desktop/t2.png')
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_thermal, None)
    kp2, des2 = sift.detectAndCompute(img_rgb, None)
    imgSift = cv2.drawKeypoints(
    img_thermal, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Match keypoints in both images
    # Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=150)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Keep good matches: calculate distinctive image features
    # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
    # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance <0.9*n.distance:
        # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    # Draw the keypoint matches between both pictures
    # Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    draw_params = dict(matchColor=(0, 0, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    keypoint_matches = cv2.drawMatchesKnn(
        img_thermal, kp1, img_rgb, kp2, matches, None, **draw_params)
    cv2.imshow("Keypoint matches", keypoint_matches)
    cv2.waitKey()
    # ------------------------------------------------------------
    # STEREO RECTIFICATION

    # Calculate the fundamental matrix for the cameras
    # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    print(inliers)
    # We select only inlier points
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]
    # Stereo rectification (uncalibrated variant)
    # Adapted from: https://stackoverflow.com/a/62607343
    #h1, w1 = img_thermal.shape
    #h2, w2 = img_rgb.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(160, 120)
    )
    # Undistort (rectify) the images and save them
    # Adapted from: https://stackoverflow.com/a/62607343
    img1_rectified = cv2.warpPerspective(img_thermal, H2, (160, 120))
    img2_rectified = cv2.warpPerspective(img_rgb, H2, (160, 120))
    cv2.imwrite("rectified_1.png", img1_rectified)
    cv2.imwrite("rectified_2.png", img2_rectified)
        #img1_rectified = cv2.warpPerspective(img_thermal, H2, (160,120))
    #img2_rectified = cv2.warpPerspective(img_rgb, H2, (160,120))
    #cv2.imwrite("rectified_1.png", img1_rectified)
    #cv2.imwrite("rectified_2.png", img2_rectified)
