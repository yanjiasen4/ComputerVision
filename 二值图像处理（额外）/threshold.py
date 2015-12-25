#coding=utf-8
import cv2 as cv2
import numpy as np
import math
import sys

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

def OTSUthreshold(hist):
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
    print "阈值为",T
    return T

def Fuzzythreshold(hist):
    first,last = getFirstLast(hist)
    T = -1
    bestV = sys.maxint
    v = 0
    mu = 0
    S = []
    W = []
    for i in range(0,first):
        S.append(-1.0)
        W.append(-1.0)
    S.append(hist[first][0])
    W.append(0)
    total = getAllPiexl(hist)
    for i in range(first,last+1):
       S.append(S[i] + hist[i][0])
       W.append(W[i] + hist[i][0]*i)

    Smu = []
    Smu.append(0)
    #记录查找表，加快计算速度
    for i in range(1,last+1):
        mu = 1/(1.0+(i+0.0)/(last-first))
        Smu.append(-mu*math.log(mu)-(1-mu)*math.log(1-mu))

    for i in range(first,last):
        v = 0
        mu = W[i]/S[i]
        for j in range(first,i+1):
            v += Smu[int(abs(j-mu))]*hist[j][0]
        if S[last]-S[i] == 0: continue
        mu = (W[last]-W[i])/(S[last]-S[i])
        for k in range(i+1,last+1):
            v += Smu[int(abs(k-mu))]*hist[k][0]
        if bestV > v:
            bestV = v
            T = i-1
    print "阈值为",T-8
    return T-8

#fileimg = cv2.imread("test.jpg")
fileimg = cv2.imread("testfromnet.png") # 读取图片
img = cv2.cvtColor(fileimg,cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([img],[0],None,[256],[0.0,255.0])
print "请选择阈值选取方式"
print "1. Otsu算法"
print "2. 模糊集算法"
select = int(input())
if select == 1:
    T = OTSUthreshold(hist)
elif select == 2:
    T = Fuzzythreshold(hist)
else:
    print "error!"
    exit()
retval, newimg=cv2.threshold(img,T,255,cv2.THRESH_BINARY)
cv2.imshow('threshold',newimg)
print "按任意键终止程序"
cv2.waitKey(0)
cv2.destroyAllWindows()
