#coding=utf-8
'''
    copyright Yang Ming, 5130379022
    Adaptive Histgram Equlaztion
'''
import cv2
import numpy as np

# calculate a histgram equlaztion mapping for a (part of) image
def calcMapping(image):
    hist = cv2.calcHist([image],[0],None,[256],[0.0,255.0])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    dst = cv2.LUT(image,cdf)
    l,w = dst.shape[:2]
    return cdf

# calculate every blocks' hist and save them into a array
def histEuqAll(image,region=8):
    tileValue = [[0 for col in range(region)] for row in range(region)]
    l,w = image.shape
    dl = l/region
    dw = w/region
    for i in range(region-1):
        for j in range(region-1):
            tileValue[i][j] = calcMapping(image[dl*i:dl*(i+1),dw*j:dw*(j+1)])
    for i in range(region-1):
        tileValue[region-1][i] = calcMapping(image[dl*(region-1):l,dw*i:dw*(i+1)])
    for j in range(region-1):
        tileValue[j][region-1] = calcMapping(image[dl*j:dl*(j+1),dw*(region-1):w])
    tileValue[region-1][region-1] = calcMapping(image[dl*(region-1):l,dw*(region-1):w])
    return tileValue

# use interpolation calculate every pixels' gray value
def interpolation(image, tileValue):
    l,w = image.shape[:2]
    region = len(tileValue)
    dl = l/region
    dw = w/region
    # center
    for i in range(0,region-1):
        for j in range(0,region-1):
            for n in range(dl):
                for m in range(dw):
                    x = dw*j+m+dw/2
                    y = dl*i+n+dl/2
                    a = float(n)/dl
                    b = float(m)/dw
                    grayValue = image[y][x]
                    image[y][x] = (1-a)*((1-b)*tileValue[i][j][grayValue]+b*tileValue[i][j+1][grayValue]) + a*((1-b)*tileValue[i+1][j][grayValue]+b*tileValue[i+1][j+1][grayValue])

    # edege
    # top and bottom
    for j in range(0,region-1):
        for n in range(dl/2):
            for m in range(dw):
                x = dw*j+m+dw/2
                y1 = n
                y2 = dl*(region-1)+n+dl/2
                b = float(m)/dw
                grayValue1 = image[y1][x]
                grayValue2 = image[y2][x]
                image[y1][x] = (1-b)*tileValue[0][j][grayValue1]+b*tileValue[0][j+1][grayValue1]
                image[y2][x] = (1-b)*tileValue[7][j][grayValue2]+b*tileValue[7][j+1][grayValue2]
        for n in range(dl/2,dl/2+l%region):
            for m in range(dw):
                x = dw*j+m+dw/2
                y = n+dl*(region-1)+dl/2
                b = float(m)/dw
                grayValue = image[y][x]
                image[y][x] = (1-b)*tileValue[7][j][grayValue]+b*tileValue[7][j+1][grayValue]

    # left and right
    for i in range(0,region-1):
        for n in range(dl):
            for m in range(dw/2):
                x1 = m
                x2 = m+dw*(region-1)+dw/2
                y = dl*i+n+dl/2
                a = float(n)/dl
                grayValue1 = image[y][x1]
                grayValue2 = image[y][x2]
                image[y][x1] = (1-a)*tileValue[i][0][grayValue1]+a*tileValue[i+1][0][grayValue1]
                image[y][x2] = (1-a)*tileValue[i][7][grayValue2]+a*tileValue[i+1][7][grayValue2]
            for m in range(dw/2,dw/2+w%region):
                x = m+dw*(region-1)+dw/2
                y = dl*i+n+dl/2
                a = float(n)/dl
                grayValue = image[y][x]
                image[y][x] = (1-a)*tileValue[i][7][grayValue]+a*tileValue[i+1][7][grayValue]

    # corner
    image[0:dl/2,0:dw/2]=cv2.LUT(image[0:dl/2,0:dw/2],tileValue[0][0])
    image[0:dl/2,(region-1)*dw+dw/2:w]=cv2.LUT(image[0:dl/2,(region-1)*dw+dw/2:w],tileValue[0][7])
    image[(region-1)*dl+dl/2:l,0:dw/2]=cv2.LUT(image[(region-1)*dl+dl/2:l,0:dw/2],tileValue[7][0])
    image[(region-1)*dl+dl/2:(region-1)*dw+dw/2:w]=cv2.LUT(image[(region-1)*dl+dl/2:(region-1)*dw+dw/2:w],tileValue[7][7])

    return image

def __main__():
    image = cv2.imread("test.png",0)
    tileValue = histEuqAll(image,8)
    image = interpolation(image, tileValue)
    cv2.imshow('ahe',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

__main__()
