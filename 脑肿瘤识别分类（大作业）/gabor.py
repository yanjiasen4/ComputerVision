#-*- coding:gbk -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi

from skimage import data
from skimage import io
from skimage.util import img_as_float
from skimage import img_as_ubyte
from skimage.filters import gabor_kernel
from skimage.filters import gabor_filter
from skimage.exposure import rescale_intensity
import os
dir_path = 'hg0005'
flair_path = 'BRATS_HG0005_FLAIR'
flair_res_path = 'BRATS_HG0005_FLAIR_GABOR_0'
t1_path = 'BRATS_HG0005_T1'
t1_res_path = 'BRATS_HG0005_T1_GABOR'
t1c_path = 'BRATS_HG0005_T1C'
testfile = 'BRATS_HG0005_FLAIR_77.png'

# Plot a selection of the filter bank kernels and their responses.
results = []

freq = input("����gabor��frequency:")
theta = input("����gabor��theta(float:�Ƕ�):")
bandwidth = input("����gabor��bandwidth(float:Ĭ��1):")
sigma_x = input("����gabor��sigma_x(float:Ĭ��None):")
sigma_y = input("����gabor��sigma_y(float:Ĭ��None):")
n_stds = input("����gabor��n_stds(scalar:Ĭ��3):")
offset = input("����gabor��offset(float:Ĭ��0):")
mode = raw_input("����gabor��mode(��constant��, ��nearest��, ��reflect��, ��mirror��, ��wrap��:Ĭ��'reflect'):")
cval = input("����gabor��cval(scalar:Ĭ��0)")
is_test = raw_input("�Ƿ����Ե���ͼƬ(Y/N):")
# Plot a selection of the filter bank kernels and their responses.

results = []
if not os.path.exists(os.path.join(dir_path,flair_res_path)):
    os.mkdir(os.path.join(dir_path,flair_res_path))
if cmp(is_test,'Y') == 0:
    tstimg = img_as_float(io.imread(os.path.join(dir_path,flair_path,testfile)))
    reresult, result = gabor_filter(tstimg,frequency=freq,theta=theta/180.0*np.pi)
    res_fname = "%s_test_%s" %(testfile[:testfile.rfind('.')],testfile[testfile.rfind('.'):])
    io.imsave(os.path.join(dir_path,flair_res_path,res_fname),rescale_intensity(reresult,out_range=(-1.,1.)))
else:
    for str_root, lst_dir, lst_file in os.walk(os.path.join(dir_path,flair_path)):
        for img_name in lst_file:
            img = img_as_float(io.imread(os.path.join(dir_path,flair_path,img_name)))
            #gabor �˲���
            reresult, result = gabor_filter(img,frequency=freq,theta=theta/180.0*np.pi)
            res_fname = "%s%s" %(img_name[:img_name.rfind('.')],img_name[img_name.rfind('.'):])
            io.imsave(os.path.join(dir_path,flair_res_path,res_fname),rescale_intensity(reresult,out_range=(-1.,1.)))
