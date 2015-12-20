'''
    copyright YangMing, 5130379022
    Fourier Transform - Computer Vision
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

# make a dft helper matrix
def dft_matrix(N):
	i,j = np.meshgrid(np.arange(N),np.arange(N))
	omega = np.exp(-2j*np.pi/N)
	w = np.power(omega,i*j)
	return w

# optimization for DFT from O(n^4) to O(n^2) use product of matrix
def dft2d(image,flags):
	h,w = image.shape[:2]
	output = np.zeros((h,w),np.complex)
	output = dft_matrix(h).dot(image).dot(dft_matrix(w))
	return output

# split result from complex to 2 real and image number
def splitComplex(image):
    h,w = image.shape[:2]
    output = np.zeros((h,w,2),np.float32)
    for i in range(h):
        for j in range(w):
            output[i][j][0] = image[i][j].real
            output[i][j][1] = image[i][j].imag
    return output

# low pass filter
def LPF(shift,image):
    rows,cols = image.shape
    crow,ccol = rows/2,cols/2
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-40:crow+40, ccol-40:ccol+40] = 1

    mshift = shift*mask
    return mshift

def HPF(shift,image):
    rows,cols = image.shape
    crow,ccol = rows/2,cols/2
    mask = np.zeros((rows,cols,2),np.uint8)
    for i in range(rows):
        for j in range(cols):
            mask[i][j] = 1
    mask[crow-40:crow+40, ccol-40:ccol+40] = 0

    mshift = shift*mask
    return mshift


def __main__():
    img = cv2.imread('test.png',0)
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft2d_img = dft2d(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_img = splitComplex(dft2d_img)

    dft_shift = np.fft.fftshift(dft_img)
    dft_ishift = HPF(dft_shift,img)
    #print dft_ishift
    dft_ishift = np.fft.ifftshift(dft_ishift)
    img_back = cv2.idft(dft_ishift)

    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

    # use plt show result

    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(img_back, cmap = 'gray')
    plt.title('Inverse DFT Image'), plt.xticks([]), plt.yticks([])
    plt.show()

__main__()
