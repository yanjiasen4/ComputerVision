from pybrain.tools.shortcuts     import buildNetwork
from pybrain.datasets            import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError
import cv2
import numpy as np

N = 66
data_width = 256
data_length = 256
data_size = data_width*data_length

dataRootPath = 'hg0005'
dataPathName_FLAIR = dataRootPath+'\BRATS_HG0005_FLAIR'
testFileName_FLAIR = '\BRATS_HG0005_FLAIR_'
dataPathName_truth = dataRootPath+'\BRATS_HG0005_truth'
testFileName_truth = '\BRATS_HG0005_truth_'
dataPathName_T1 = dataRootPath+'\BRATS_HG0005_T1'
testFileName_T1 = '\BRATS_HG0005_T1_'

def read_data(fileDir,filePre):
    data = np.zeros((N,data_width,data_length),dtype=np.uint8)
    for i in range(N):
        fileIndex = str(i + 67) + '';
        fileFullName = fileDir+filePre+fileIndex+'.png'
        print fileFullName
        tmp = cv2.imread(fileFullName,0)
        data[i] = tmp
    return data

def getImage(a,w,l):
    output = np.zeros((w,l),dtype=np.uint8)
    for i in range(w):
        for j in range(l):
            output[i][j] = a[i*w+j]
    return output

test1 = read_data(dataPathName_FLAIR,testFileName_FLAIR)
target = read_data(dataPathName_truth,testFileName_truth)
net = buildNetwork(data_size, 10, data_size)
ds = SupervisedDataSet(data_size, data_size)
for i in range(N):
    ds.addSample(np.ravel(test1[i]),np.ravel(target[i]))
trainer = BackpropTrainer(net, ds)
for i in range(10):
    trainer.trainEpochs(1)

testImg = getImage(net.activate(np.ravel(test1[50])),data_length,data_width)
cv2.imshow("test",testImg)
cv2.waitKey(0)
cv2.destroyAllWindows();
