# ComputerVision

## 开发环境

* 辅助图形处理库：[opencv3.0](http://opencv.org/ "opencv")  
* 编程语言：[python2.7](https://www.python.org/ "python")  

### 说明
由于opencv3.0对SIFT算子的支持在额外仓库中，而我尝试了加入额外的modules在vs2015下make没有成功  
随后我将opencv版本降为了2.4.10，并且自己实现了drawMatchesKnn函数，所以在SIFT中我使用的是opencv2.4.10 
