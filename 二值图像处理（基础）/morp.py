#coding=utf-8
import copy

class BinaryMartix(object):
    def __init__(self,filename):
        self.martix = []
        self.height = self.width = 0
        input = open(filename,'r')
        for line in input:
            tmp = []
            for bit in line:
                if bit =='1': tmp.append(1)
                else: tmp.append(0)
            self.martix.append(tmp)
            self.width = len(tmp)
            self.height += 1
        input.close()

    def printMartix(self):
        for i in range(0,self.height):
            for j in range(0,self.width):
                print self.martix[i][j],
            print ''

    def doDilation(self,x,y,structuring):
        w = len(structuring[0])
        h = len(structuring)
        x_offset = (w-1)/2
        y_offset = (h-1)/2
        for i in range(0,h):
            for j in range(0,w):
                if structuring[i][j] == 1 and self.martix[y-y_offset+i][x-x_offset+j] == 1: return True
                else: continue
        return False

    def doErosion(self,x,y,structuring):
        w = len(structuring[0])
        h = len(structuring)
        x_offset = (w-1)/2
        y_offset = (h-1)/2
        for i in range(0,h):
            for j in range(0,w):
                if structuring[i][j] != self.martix[y-y_offset+i][x-x_offset+j]: return False
                else: continue
        return True

    def dilation(self,structuring):
        output = copy.deepcopy(self.martix)
        s_width = len(structuring[0])
        s_height= len(structuring)
        for i in range(s_height/2,self.height-s_height/2):
            for j in range(s_width/2,self.width-s_width/2):
                if self.doDilation(j,i,structuring):
                    output[i][j] = 1
                else:
                    output[i][j] = 0
        return output

    def erosion(self,structuring):
        output = copy.deepcopy(self.martix)
        s_width = len(structuring[0])
        s_height= len(structuring)
        for i in range(s_height/2,self.height-s_height/2):
            for j in range(s_width/2,self.width-s_width/2):
                if self.doErosion(j,i,structuring):
                    output[i][j] = 1
                else:
                   output[i][j] = 0
        return output

def printMartix(martix):
    h = len(martix)
    w = len(martix[0])
    for i in range(0,h):
        for j in range(0,w):
            print martix[i][j],
        print ''

def __main__():
    bmd = BinaryMartix('tmp1.txt')
    bme = BinaryMartix('tmp1.txt')
    str_element = [[1,1,1]]
    printMartix(bmd.dilation(str_element))
    print ''
    printMartix(bme.erosion(str_element))


__main__()
