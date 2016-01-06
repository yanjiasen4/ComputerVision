from pybrain.tools.shortcuts     import buildNetwork
from pybrain.datasets            import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError
import cv2
import numpy as np

N = 66
data_width = 256
data_length = 256
block_width = 8
block_length = 8
data_size = data_width*data_length
block_size = block_width*block_length

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
        print 'read file: ' + fileFullName
        tmp = cv2.imread(fileFullName,0)
        data[i] = tmp
    return data

def mat2str(mat):
    h,l = mat.shape[:2]
    res = '[\n'
    for i in range(h):
        for j in range(l):
            res += (str(mat[i][j]) + ',')
        res += '\n'
    for j in range(l):
        res += '  '
    res += ']\n'
    return res

def getImage(a,w,l):
    output = np.zeros((w,l),dtype=np.uint8)
    for i in range(w):
        for j in range(l):
            output[i][j] = a[i*w+j]
    return output

# img: 256x256
def trainByImg(net, test, target):
    h,l = test.shape[:2]
    for i in range(h):
        for j in range(l):
            if target[i,j] != 0:
                target[i,j] = 1

    ds = SupervisedDataSet(block_size, block_size)
    for i in range(0,data_length,block_width):
        for j in range(0,data_width,block_length):
            ds.addSample(np.ravel(test[i:i+block_length,j:j+block_width]), np.ravel(target[i:i+block_length,j:j+block_width]))

    trainer = BackpropTrainer(net, ds)
    for i in range(1):
        trainer.trainEpochs(1)

def activateByImg(net, inputTest):
    output = np.zeros((data_length,data_width), dtype=np.int)
    fileObj = open('res.txt','a')
    for i in range(0,data_width,block_width):
        for j in range(0,data_length,block_length):
            output[i:i+block_length,j:j+block_width] = (net.activate(np.ravel(inputTest[i:i+block_length,j:j+block_width]))).reshape(block_length,block_width)
            buffer_str = mat2str(output[i:i+block_length,j:j+block_width])
            fileObj.write(buffer_str)
    fileObj.close()
    h,l = output.shape[:2]
    for i in range(h):
        for j in range(l):
            if output[i][j] != 0:
                print '!'
                output[i][j] = 255
    return output

def usage():
    print 'input a integer in [0,65] to get result of a input'
    print 'input q to exit'


# main
def __main__():
    tests = read_data(dataPathName_FLAIR,testFileName_FLAIR)
    targets = read_data(dataPathName_truth,testFileName_truth)
    '''
    net = buildNetwork(data_size, 10, data_size)
    ds = SupervisedDataSet(data_size, data_size)
    for i in range(N):
        ds.addSample(np.ravel(test1[i]),np.ravel(target[i]))
    trainer = BackpropTrainer(net, ds)
    for i in range(10):
        trainer.trainEpochs(1)
    testImg = getImage(net.activate(np.ravel(test1[50])),data_length,data_width)
    '''
    net = buildNetwork(block_size, 16, block_size)
    for i in range(N):
        print 'train: '+str(i)
        trainByImg(net, tests[i], targets[i])

    usage()
    flag = True
    while flag:
        ipt = raw_input('>>')
        if ipt.isalpha():
            if ipt=='q':
                print 'quit'
                flag = False
            else:
                print 'error input: [%s]',ipt
        elif ipt.isdigit():
            num = int(ipt)
            if num >= 0 and num <= 65:
                res = activateByImg(net, tests[num])
                cv2.imshow("test",res)
                cv2.waitKey(0)
                cv2.destroyAllWindows();
            else:
                print('error index of tests [%d], should in [0,65]',num)

__main__()
