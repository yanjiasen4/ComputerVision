#coding=utf-8
import cv2
import numpy as np
import random

"""
    I couldn't find drawMatches() in OpenCV 2.4.10
    while 3.0 don't support SIFT
    modify function drawMatches() from rayryeng's answer at
    http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python
"""
def drawMatchesKnn(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # get random color
        randR = random.uniform(0,255)
        randG = random.uniform(0,255)
        randB = random.uniform(0,255)
        randColor = (randB,randG,randR)
        # Get the matching keypoints for each of the images
        for ind in mat:
            img1_idx = ind.queryIdx
            img2_idx = ind.trainIdx

            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt

            cv2.circle(out, (int(x1),int(y1)), 4, randColor, 1)
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, randColor, 1)

            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), randColor, 1)

    # Show the image
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyWindow('Matched Features')
    return out

'''select your input img'''
#img1 = cv2.imread('test1.png')
#img2 = cv2.imread('test2.png')
img1 = cv2.imread('Pic1-1.png')
img2 = cv2.imread('Pic1-2.png')
#img1 = cv2.imread('Pic2-1.png')
#img2 = cv2.imread('Pic2-2.png')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT(500,5)
cv2.SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()


matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = drawMatchesKnn(gray1,kp1,gray2,kp2,good)

# save the result img
cv2.imwrite('sift_matches_Pic2.jpg',img3)
