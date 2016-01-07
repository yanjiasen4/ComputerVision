
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.datasets            import SupervisedDataSet
from pybrain.structure.modules   import SigmoidLayer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets            import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError

#from sklearn import datasets, svm, metrics
import cv2
import numpy as np
import random

N = 66
data_width = 256
data_length = 256
block_width = 2
block_length = 2
data_size = data_width*data_length
block_size = (2*block_width+1)*(2*block_length+1)

dataRootPath = 'hg0005'
dataPathName_FLAIR = dataRootPath+'\BRATS_HG0005_FLAIR_GABOR'
testFileName_FLAIR = '\BRATS_HG0005_FLAIR_'
dataPathName_truth = dataRootPath+'\BRATS_HG0005_truth'
testFileName_truth = '\BRATS_HG0005_truth_'
dataPathName_T1 = dataRootPath+'\BRATS_HG0005_T1'
testFileName_T1 = '\BRATS_HG0005_T1_'
testPathName_FLAIR = dataRootPath+'\BRATS_HG0005_FLAIR'

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
def trainByImg(net, test, target, trainNum):
    h,l = test.shape[:2]
    for i in range(h):
        for j in range(l):
            if target[i,j] != 0:
                target[i,j] = 255

    ds = SupervisedDataSet(data_size, 10, data_size)
    #ds = ClassificationDataSet(block_size, nb_classes=2, class_labels=[0,255])
    '''
    for i in range(trainNum):
        x = random.uniform(block_width, data_width-block_width-1)
        y = random.uniform(block_length,data_length-block_length-1)
        if target[y,x] == 255:
            targetValue = 1
        else:
            targetValue = 0
        ds.addSample(np.ravel(test[y-block_length:y+block_length+1,x-block_width:x+block_width+1]), [targetValue])

    for i in range(0,data_length,block_width):
        for j in range(0,data_width,block_length):
            ds.addSample(np.ravel(test[i:i+block_length,j:j+block_width]), np.ravel(target[i:i+block_length,j:j+block_width]))

    #ds._convertToOneOfMany()
    '''
    #ds.addSample(np.ravel(test), np.ravel(target))
    trainer = BackpropTrainer(net, ds)
    for i in range(1):
        trainer.trainEpochs(1)


def activateByImg(net, inputTest):
    output = np.zeros((data_length,data_width), dtype=np.uint8)
    '''
    for i in range(block_length,data_length-block_length):
        for j in range(block_width,data_width-block_width):
            tmp = net.activate(np.ravel(inputTest[i-block_length:i+block_length+1,j-block_width:j+block_width+1]))
            print tmp
            #print tmp.argmax(axis=0)
            if tmp[0] >= 0.9:
                output[i,j] = 255
    '''
    output = net.activate(np.ravel(inputTest)).reshape(output.shape)
    return output

def predictByImg(classifier, img):
    output = np.zeros((data_length,data_width), dtype=np.uint8)
    ipt = []
    for i in range(block_length,data_length-block_length):
        for j in range(block_width,data_width-block_width):
            ipt.append((np.ravel(img[i-block_length:i+block_length+1,j-block_width:j+block_width+1])))
    tmp = classifier.predict(ipt).reshape(data_length-2*block_length,data_width-2*block_width)
    for i in range(block_length,data_length-block_length):
        for j in range(block_width,data_width-block_width):
            output[i,j] = tmp[i-block_length,j-block_width]
    fileObj = open('log.txt','w')
    str_buffer = mat2str(output)
    fileObj.write(str_buffer)
    fileObj.close()
    return output

def usage():
    print 'input a integer in [0,65] to get result of a input'
    print 'input q to exit'


# main
def __main__():
    tests = read_data(testPathName_FLAIR,testFileName_FLAIR)
    '''
    tests0 = read_data(dataPathName_FLAIR+'_0',testFileName_FLAIR)
    tests1 = read_data(dataPathName_FLAIR+'_45',testFileName_FLAIR)
    tests2 = read_data(dataPathName_FLAIR+'_90',testFileName_FLAIR)
    tests3 = read_data(dataPathName_FLAIR+'_135',testFileName_FLAIR)
    '''
    targets = read_data(dataPathName_truth,testFileName_truth)
    #ds = ClassificationDataSet(block_size, class_labels=[0,255])
    #train = []
    #result = []
    for k in range(N):
        '''
        test0 = tests0[k]
        test1 = tests1[k]
        test2 = tests2[k]
        test3 = tests3[k]
        '''
        target = targets[k]
        '''
        h,l = target.shape[:2]
        for i in range(h):
            for j in range(l):
                if target[i,j] != 0:
                    target[i,j] = 255
        for i in range(data_size/20):
            x = random.uniform(block_width, data_width-block_width-1)
            y = random.uniform(block_length,data_length-block_length-1)
            if target[y,x] == 255:
                targetValue = 1
            else:
                targetValue = 0
            flag = 0
            if test0[y-block_length:y+block_length+1,x-block_width:x+block_width+1].any() != 0:
                flag = 1
            if flag == 1:
                ds.addSample(np.ravel(test0[y-block_length:y+block_length+1,x-block_width:x+block_width+1]), [targetValue])
                ds.addSample(np.ravel(test1[y-block_length:y+block_length+1,x-block_width:x+block_width+1]), [targetValue])
                ds.addSample(np.ravel(test2[y-block_length:y+block_length+1,x-block_width:x+block_width+1]), [targetValue])
                ds.addSample(np.ravel(test3[y-block_length:y+block_length+1,x-block_width:x+block_width+1]), [targetValue])
                #train.append(np.ravel(test0[y-block_length:y+block_length+1,x-block_width:x+block_width+1]))
                #train.append(np.ravel(test1[y-block_length:y+block_length+1,x-block_width:x+block_width+1]))
                #train.append(np.ravel(test2[y-block_length:y+block_length+1,x-block_width:x+block_width+1]))
                #train.append(np.ravel(test3[y-block_length:y+block_length+1,x-block_width:x+block_width+1]))
                #result.append(targetValue)

    ds._convertToOneOfMany()

    print "Number of training patterns: ", len(ds)
    print "Input and output dimensions: ", ds.indim, ds.outdim
    print "First sample (input, target, class):"
    print ds['input'][0], ds['target'][0], ds['class'][0]
    '''
    ds = SupervisedDataSet(data_size, data_size)
    for k in range(N):
        ds.addSample(np.ravel(tests[k]), np.ravel(targets[k]))
    net = buildNetwork(ds.indim, 10, ds.outdim)
    trainer = BackpropTrainer(net, ds, momentum=0.1, verbose=True, weightdecay=0.01)
    for i in range(20):
        print 'epochs '+str(i),
        trainer.trainEpochs(1)


    '''
    # support vector machine
    # but it's too slow
    classifier = svm.SVC(gamma=0.001)
    print 'trainning...'
    classifier.fit(train, result)
    '''

    usage()
    flag = True
    kernel = np.ones((5,5),np.uint8)
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
            if num >= 0 and num < N:
                #res = predictByImg(classifier, tests[num])
                res = activateByImg(net, tests[num])
                res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
                cv2.imshow("test",res)
                cv2.waitKey(0)
                cv2.destroyAllWindows();
            else:
                print('error index of tests [%d], should in [0,65]',num)

__main__()
