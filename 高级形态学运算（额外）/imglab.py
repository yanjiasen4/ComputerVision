#coding=utf-8
array = [0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,1,\
         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,\
         0,0,1,1,0,0,1,1,1,1,0,1,1,1,0,1,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,\
         1,1,0,0,1,1,0,0,1,1,0,1,1,1,0,0,\
         1,1,0,0,1,1,1,0,1,1,0,0,1,0,0,0]

def Vthin(img):
    h = len(img)
    w = len(img[0])
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = img[i,j-1]+img[i,j]+img[i,j+1] if 0<j<w-1 else 1
                if img[i,j] == 0 and M != 0 :
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and img[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    img[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return img

def Hthin(img):
    h = len(img)
    w = len(img[0])
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = img[i-1,j]+img[i,j]+img[i+1,j] if 0<i<h-1 else 1   
                if img[i,j] == 0 and M != 0:         
                    a = [0]*9
                    for k in range(3):
                        for l in range(3):
                            if -1<(i-1+k)<h and -1<(j-1+l)<w and img[i-1+k,j-1+l]==255:
                                a[k*3+l] = 1
                    sum = a[0]*1+a[1]*2+a[2]*4+a[3]*8+a[5]*16+a[6]*32+a[7]*64+a[8]*128
                    img[i,j] = array[sum]*255
                    if array[sum] == 1:
                        NEXT = 0
    return img

def IndexThin(img,num=10):
    for i in range(num):
        Vthin(img)
        Hthin(img)
    return img
