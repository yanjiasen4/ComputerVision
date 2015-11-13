#coding=utf-8
import cv2
import numpy as np
import copy
import imglab # a thinning algorithm from internet

ele1 = [[255,255,255],[-1,0,-1],[0,0,0]]
ele2 = [[-1,255,255],[0,0,255],[0,0,-1]]
ele3 = [[0,-1,255],[0,0,255],[0,-1,255]]
ele4 = [[0,0,-1],[0,0,255],[-1,255,255]]
ele5 = [[0,0,0],[-1,0,-1],[255,255,255]]
ele6 = [[-1,0,0],[255,0,0],[255,255,-1]]
ele7 = [[255,-1,0],[255,0,0],[255,-1,0]]
ele8 = [[255,255,-1],[255,0,0],[-1,0,0]]

# return height,length of img
def getImgSize(img):
    return len(img),len(img[0])

def IsA(array,i,j):
    tmp = [array[i][j],array[i-1][j],array[i-1][j+1],array[i][j+1],array[i+1][j+1],array[i+1][j],array[i+1][j-1],array[i][j-1],array[i-1,j-1]]
    n=0
    for i in range(1,8):
        if tmp[i]==0 and tmp[i+1]==255:
            n = n+1
    return n

# Zhang-thinning algorithmn
def Zhangthin(img):
    mark=1
    height = len(img)
    length = len(img[0])
    while mark==1:
        mark=0
        for i in range(1,height-1):
            for j in range(1,length-1):
                cond = 0
                if img[i][j]==255:
                    n = 0
                    for ii in range(-1,2):
                        for jj in range(-1,2):
                            n = n+(img[i+ii][j+jj])/255
                    if n >=3 and n <= 7:
                        cond += 1
                    if IsA(img,i,j)==1:
                        cond += 1
                    if (int(img[i-1][j])*img[i][j+1]*img[i+1][j])==0:
                        cond += 1
                    if (int(img[i][j+1])*img[i+1][j]*img[i][j-1])==0:
                        cond += 1
                    if cond == 4:
                        mark = 1
                        img[i][j] = 0

        for i in range(1,height-1):
            for j in range(1,length-1):
                cond = 0
                if img[i][j]==255:
                    n = 0
                    for ii in range(-1,2):
                        for jj in range(-1,2):
                            n = n + img[i+ii][j+jj]
                    if n>=3 and n<=7:
                        cond += 1
                    if IsA(img,i,j)==1:
                        cond += 1
                    if (int(img[i-1][j])*img[i][j+1]*img[i][j-1])==0:
                        cond += 1
                    if (int(img[i][j-1])*img[i+1][j]*img[i][j-1])==0:
                        cond += 1
                    if cond == 4:
                        mark = 1
                        img[i][j] = 0

        return img

# fill the img's edge with white color
def edgeFilling(img):
    height = len(img)
    length = len(img[0])
    for i in range(0,length):
        img[0][i] = 255
        img[height-1][i] = 255
    for j in range(0,height):
        img[j][0] = 255
        img[j][length-1] = 255
    return img

# hit-or-miss operation                  
def homTransform(img,ele):
    height = len(img)
    length = len(img[0])
    h_off = len(ele)/2
    l_off = len(ele[0])/2
    source = copy.deepcopy(img)
    for i in range(h_off,height-h_off):
        for j in range(l_off,length-l_off):
            flag = 0
            for ii in range(-h_off,h_off+1):
                if flag == 1: break
                for jj in range(-l_off,l_off+1):
                    imgbit = source[i+ii][j+jj]
                    img[i][j] = 0
                    eletmp = ele[h_off+ii][l_off+jj]
                    if eletmp == -1: continue
                    elif eletmp != imgbit:
                        flag = 1
                        img[i][j] = 255
                        break
    return img

#diff and union operate 
#src and dest must be the same size
#diff
def diffImg(src,tmp):
    height,length = getImgSize(src)
    for i in range(0,height):
        for j in range(0,length):
            if src[i][j] == 0:
                if tmp[i][j] == 0:
                    src[i][j] = 255
    return src

#union
def unionImg(src,tmp):
    heigth,length = getImgSize(src)
    for i in range(0,height):
        for j in range(0,length):
            if tmp[i][j] == 0:
                if src[i][j] != 0:
                    src[i][j] = 0
    return src

# thinning by a single structuring element
def ThinningByEle(img,ele):
    source = copy.deepcopy(img)
    img = homTransform(img,ele)
    return diffImg(source,img)

# thickening by a single structuring element
def ThickeningByEle(img,ele):
    source = copy.deepcopy(img)
    img = homTransform(img,ele)
    return unionImg(source,img)

# thinning by a list of structuring elements
def Thinning(img,eleList):
    t = len(eleList)
    for i in range(0,t):
        img = ThinningByEle(img,eleList[i])
        img = edgeFilling(img)
    return img

# thickening by a list of structuring elements
def Thickening(Img,eleList):
    t = len(eleList)
    for i in range(0,t):
        img = ThickeningByEle(img,eleList[i])
        img = edgeFilling(img)
    return img
                          
fileimg = cv2.imread('test2.bmp')
img = cv2.cvtColor(fileimg,cv2.COLOR_BGR2GRAY)
retval, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
img = cv2.bitwise_not(img)
#eleList = [ele1,ele2,ele3,ele4,ele5,ele6,ele7,ele8]
#res = Thinning(img,eleList)
#for i in range(80):
#    res = Thinning(res,eleList)
eleList = [ele1]
res = Thinning(img,eleList)
cv2.imshow("result",res)
cv2.waitKey(0)
cv2.destroyAllWindows() 
