import cv2
import numpy as np

# 直方图均衡算法
def histEqu(img): 
    hist = cv2.calcHist([img],[0],None,[256],[0.0,255.0])
    cdf = hist.cumsum()  
    print cdf
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    dst = cv2.LUT(img,cdf)
    return dst

# 生成图片直方图图片
def histShow(img):
    hist = cv2.calcHist([img],[0],None,[256],[0.0,255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)  
    histImg = np.zeros([256,256], np.uint8)
    hpt = int(0.9* 256)
    color = [255,255,255]

    for i in range(256):
        intensity = int(hist[i]*hpt/maxVal)
        cv2.line(histImg, (i,256), (i,256-intensity), color)

    return histImg

def __main__():
    img = cv2.imread("test.png")
    dst = histEqu(img)
    histImg = histShow(img)
    histDst = histShow(dst)
    cv2.imshow("imgHist",histImg)
    cv2.imshow("histDst",histDst)
    cv2.imshow("result",dst)
    cv2.imwrite("result.png",dst,[int(cv2.IMWRITE_PNG_COMPRESSION), 0]) #保存结果图片
    cv2.waitKey(0)
    cv2.destroyAllWindows()

__main__()
