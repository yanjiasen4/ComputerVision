#coding=utf-8
'''
Copyright, Yang Ming
This file implement the canny edge detector which contains 4 steps
1. Gaussian Blur --- use cv2.GaussianBlur()
2. Compute magnitude of gradient --- a simple unitary
     -1, 1   -1, 0
      0, 0    1, 0
3. Non-maxima suppression --- use difference algorithm to calculate every piexl's 8-neighbors piexls
4. Hysteresis thresholding --- use Otsu algorithm
'''
import cv2
import numpy as np
import copy
import math

'''
    g1 g2 g3
    g4    g5
    g6 g7 g8
'''

WHITE = 255
BLACK = 0

def hasEdgeNeighbor(i,j,img):
    g1 = img[i-1][j-1]
    g2 = img[i-1][j]
    g3 = img[i-1][j+1]
    g4 = img[i][j-1]
    g5 = img[i][j+1]
    g6 = img[i+1][j-1]
    g7 = img[i+1][j]
    g8 = img[i+1][j+1]
    if g1==WHITE \
    or g2==WHITE \
    or g3==WHITE \
    or g4==WHITE \
    or g5==WHITE \
    or g6==WHITE \
    or g7==WHITE \
    or g8==WHITE:
        return True
    else:
        return False

def isConnected(i,j,img):
    g1 = img[i-1][j-1]
    g2 = img[i-1][j]
    g3 = img[i-1][j+1]
    g4 = img[i][j-1]
    g5 = img[i][j+1]
    g6 = img[i+1][j-1]
    g7 = img[i+1][j]
    g8 = img[i+1][j+1]
    if g1+g5*g8*g7 == 0:
        return True
    if g2+g4*g5*g6*g7*g8 == 0:
        return True
    if g3+g7*g6*g4 == 0:
        return True
    if g4+g2*g3*g5*g7*g8 == 0:
        return True
    if g5+g1*g2*g4*g6*g7 == 0:
        return True
    if g6+g2*g3*g5 == 0:
        return True
    if g7+g4*g1*g2 == 0:
        return True
    if g8+g1*g2*g3*g4*g5 == 0:
        return True
    return False

def cannyDetector(img):
    #binaryzation
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #gaussian blur
    img = cv2.GaussianBlur(img, (3,3), 0)
    dimg = copy.deepcopy(img)
    height = len(img)
    width = len(img[0])
    dx = [[0 for col in range(width)] for row in range(height)]
    dy = [[0 for col in range(width)] for row in range(height)]
    for i in range(0,height-1):
        for j in range(0,width-1):
            dx[i][j] = float(img[i][j+1]) - img[i][j]
            dy[i][j] = float(img[i+1][j]) - img[i][j]
            dimg[i][j] = math.sqrt(dx[i][j]*dx[i][j] + dy[i][j]*dy[i][j])
    
    #non-maxima suppression
    kimg = copy.deepcopy(dimg)
    for i in range(1,height-1):
        for j in range(1,width-1):
            if dimg[i][j] == BLACK:
                kimg[i][j] = BLACK
            else:
                gradX = dx[i][j]
                gradY = dy[i][j]
                gradT = dimg[i][j]
                if abs(gradY) > abs(gradX):
                    w = abs(gradX) / abs(gradY)
                    g2 = dimg[i-1][j]
                    g4 = dimg[i+1][j]
                    if gradX*gradY > 0:
                        g1 = dimg[i-1][j-1]
                        g3 = dimg[i+1][j-1]
                    else:
                        g1 = dimg[i-1][j+1]
                        g3 = dimg[i+1][j-1]
                else:
                    w = abs(gradY) / abs(gradX)
                    g2 = dimg[i][j-1]
                    g4 = dimg[i][j+1]
                    if gradX*gradY > 0:
                        g1 = dimg[i+1][j+1]
                        g3 = dimg[i-1][j-1]
                    else:
                        g1 = dimg[i-1][j+1]
                        g3 = dimg[i+1][j-1]
                gradTemp1 = w*g1 + (1-w)*g2
                gradTemp2 = w*g3 + (1-w)*g4
                if gradT >= gradTemp1 and gradT >= gradTemp2:
                    kimg[i][j] = gradT
                else:
                    kimg[i][j] = BLACK

    
    for i in range(height):
        kimg[i][0] = BLACK
        kimg[i][width-1] = BLACK
    for j in range(width):
        kimg[0][j] = BLACK
        kimg[height-1][j] = BLACK

    T = OTSUthreshold(kimg)
    low_threshold = T/3
    high_threshold = low_threshold*2
    timg = copy.deepcopy(kimg)
    edgel = [[0 for col in range(width)] for row in range(height)]
    edgeh = [[0 for col in range(width)] for row in range(height)]
    for i in range(0,height):
        for j in range(0,width):
            if kimg[i][j] < low_threshold:
                timg[i][j] = BLACK
            elif kimg[i][j] > high_threshold:
                timg[i][j] = WHITE
                edgeh[i][j] = WHITE
            else:
                timg[i][j] = BLACK
                edgel[i][j] = WHITE

    
    for i in range(1,height-1):
        for j in range(1,width-1):
            if edgel[i][j] == WHITE:
                if hasEdgeNeighbor(i,j,edgeh):
                #if isConnected(i,j,edgeh):
                    timg[i][j] = WHITE
    
    return timg

def getAllPiexl(hist):
    total = 0
    for i in range(0,256):
        total += hist[i][0]
    return total

def getFirstLast(hist):
    for i in range(0,256):
        if hist[i][0] != 0:
            first = i
            break
    for j in range(255,first,-1):
        if hist[j][0] != 0:
            last = j
            break
    return first,last

def OTSUthreshold(img):
    hist = cv2.calcHist([img],[0],None,[256],[0.0,255.0])
    T = 0
    maxV = 0
    v = 0
    total = getAllPiexl(hist)
    first,last = getFirstLast(hist)
    for i in range(first,last):
        if hist[i][0] == 0: continue
        t1 = 0
        t2 = 0
        s1 = 0
        s2 = 0
        for j in range(first,i+1):
            t1 += hist[j][0]*j
            s1 += hist[j][0]
        for k in range(i+1,last+1):
            t2 += hist[k][0]*k
            s2 += hist[k][0]    
        t1 = t1/s1
        t2 = t2/s2
        w1 = s1/total
        w2 = 1-w1
        v = w1*w2*(t1-t2)*(t1-t2)
        if maxV < v:
            maxV = v
            T = i
    print T
    return T

def __main__():
    img = cv2.imread("test.png")
    canny = cannyDetector(img)
    cannydemo = cv2.Canny(img, 50, 150)
    cv2.imshow('Canny', canny)
    # use cv2's canny detector to compare with my implement
    cv2.imshow('Cannydemo', cannydemo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

__main__()
