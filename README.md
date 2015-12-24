# ComputerVision

## 项目介绍

我在计算机视觉课程上的作业仓库，包括以下算法  
* [腐蚀、膨胀操作](https://github.com/yanjiasen4/ComputerVision/tree/master/%E4%BA%8C%E5%80%BC%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%EF%BC%88%E5%9F%BA%E7%A1%80%EF%BC%89)
* [Otsu阈值算法](https://github.com/yanjiasen4/ComputerVision/tree/master/%E4%BA%8C%E5%80%BC%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86%EF%BC%88%E9%A2%9D%E5%A4%96%EF%BC%89)
* [骨骼提取](https://github.com/yanjiasen4/ComputerVision/tree/master/%E9%AB%98%E7%BA%A7%E5%BD%A2%E6%80%81%E5%AD%A6%E8%BF%90%E7%AE%97%EF%BC%88%E9%A2%9D%E5%A4%96%EF%BC%89)
* [直方图均衡化](https://github.com/yanjiasen4/ComputerVision/tree/master/%E7%9B%B4%E6%96%B9%E5%9B%BE%E5%9D%87%E8%A1%A1%EF%BC%88%E5%9F%BA%E7%A1%80%EF%BC%89)
* [canny边缘提取](https://github.com/yanjiasen4/ComputerVision/tree/master/%E8%BE%B9%E7%BC%98%E6%8F%90%E5%8F%96)
* [SIFT特征提取](https://github.com/yanjiasen4/ComputerVision/tree/master/SIFT%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96)
* [傅里叶小波变换](https://github.com/yanjiasen4/ComputerVision/tree/master/%E5%82%85%E9%87%8C%E5%8F%B6%E5%B0%8F%E6%B3%A2%E5%8F%98%E6%8D%A2)
* [自适应直方图均衡化](https://github.com/yanjiasen4/ComputerVision/tree/master/%E8%87%AA%E9%80%82%E5%BA%94%E7%9B%B4%E6%96%B9%E5%9B%BE%E5%9D%87%E8%A1%A1%EF%BC%88%E9%A2%9D%E5%A4%96%EF%BC%89)

## 开发环境

* 辅助图形处理库：[opencv3.0](http://opencv.org/ "opencv3.0")/[opencv2.4](http://opencv.org/ "opencv2.4")
* 编程语言：[python2.7](https://www.python.org/ "python2.7")  

## 说明

由于opencv3.0对SIFT算子的支持在额外仓库中，而我尝试了加入额外的modules在vs2015下make没有成功  
随后我将opencv版本降为了2.4.10，并且自己实现了drawMatchesKnn函数，所以在SIFT中我使用的是opencv2.4.10 
