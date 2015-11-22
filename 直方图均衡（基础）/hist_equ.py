import cv2
import numpy as np
import math

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


def histEqu(img):
    width = len(img[0])
    length = len(img)
    hist = cv2.calcHist([img],[0],None,[256],[0.0,255.0])
    lut = np.zeros(256, dtype = img.dtype)
    f,l = getFirstLast(hist)
    step = f-l+1
    p_hist = []
    s_hist = []
    for i,v in enumerate(lut):
        if i < f:
            lut[i] = 255
        elif i > l:
            lut[i] = 0
        else:
            lut[i] = int(255-255.0*(i-f)/step-0.5)
    return cv2.LUT(img,lut)
        

def __main__():
    img = cv2.imread("test.png")
    dst = histEqu(img)
    cv2.imshow("result",dst)
    cv2.imwrite("result.png",dst,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

__main__()
