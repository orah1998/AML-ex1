import numpy as np
import random

#getting data:
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
# read data
mnist = fetch_mldata("MNIST original", data_home="./data")
X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
x = [ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
y = [ey for ey in Y if ey in [0, 1, 2, 3]]
# suffle examples
x, y = shuffle(x, y, random_state=1)


#setting gloabal variables
tags=[0,1,2,3]
fi=0.001
myLamda=0.0001
epoches=20
#x = 784X1




#@@@@@@@@@@@@@@@@@@@@@@@@@@ Loss Functions @@@@@@@@@@@@@@@@@@@@@@@@2
def surrogateLoss(val):
    return max([0,1-val])


def lossDecoding(vec,matrix,ex):
    min=np.inf
    ans=0.0
    for r,line in enumerate(matrix):
        sum=0
        for s,pred in enumerate(vec):
            sum+=surrogateLoss(matrix[r][s]*np.dot(svms[s].w,ex))

        if(min>sum):
            min=sum
            ans=r

    return ans



def lossPred(svms,testing_x,testing_y,matrix):    
    counter=0
    for i,ex in enumerate(testing_x):
        vec=[]
        for svm in svms:
            vec.append(svm.singlePredict(ex))

        if(float(lossDecoding(vec,matrix,ex))==float(testing_y[i])):
            counter+=1

    print("accuracy is: "+str((float(counter)/len(testing_y))*100)+"%")    





#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Hamming functions @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def hammingPred(svms,testing_x,testing_y,matrix):    
    counter=0
    for i,ex in enumerate(testing_x):
        vec=[]
        for svm in svms:
            vec.append(svm.singlePredict(ex))

        if(float(hammingDecoding(vec,matrix))==float(testing_y[i])):
            counter+=1

    print("accuracy is: "+str((float(counter)/len(testing_y))*100)+"%")    





def hammingDecoding(vec,matrix):
    min=np.inf
    ans=0.0
    for r,line in enumerate(matrix):
        sum=0
        for s,pred in enumerate(vec):
            sum+=(1-sign(matrix[r][s]*pred))/2

        if(min>sum):
            min=sum
            ans=r

    return ans







#@@@@@@@@@@@@@@@@@@@@@@@@ CHANGING FUNCTIONS @@@@@@@@@@@@@@@@@@@@@@@@@@@@



#changing y to match the current binary classification (One VS. All)
def change(label, my_y):
    tempY=my_y.copy()
    for i,tag in enumerate(tempY):
        if(tag==label):
            tempY[i]=1
        else:
            tempY[i]=-1

    return tempY




def changeAllPairs(label1,label2,my_y,my_x):
    tempY=my_y.copy()
    tempX=my_x.copy()
    for i,tag in enumerate(tempY):
        if(tag==label1):
            tempY[i]=1
        elif(tag==label2):
            tempY[i]=-1
        else:
            #tempX.pop(i)
            #tempY.pop(i)
            tempY[i]=0

    return tempX,tempY




#function for random matrix
#ci is -1 0 or 1 according to random matrix
def changeRandom(my_y, c0, c1, c2, c3):
    tempY=my_y.copy()
    for i,tag in enumerate(tempY):
        if(tag==0.0):
            tempY[i]=c0
        if(tag==1.0):
            tempY[i]=c1
        if(tag==2.0):
            tempY[i]=c2
        if(tag==3.0):
            tempY[i]=c3

    return tempY







#getting the sign from single SVM prediction
def sign(num):
    if(num>0):
        return 1
    if(num==0):
        return 0

    return -1
    




class SVM(object):

    def __init__(self,pi,lamda,epoch):
        self.w=np.zeros((1,784))
        self.Pi=pi
        self.lamda=lamda
        self.epoch=epoch
        

    def execute(self,x,y):
        beginPi=self.Pi
        for t in range(1,self.epoch):
            self.Pi=beginPi/np.sqrt(t)
            for pic,tag in zip(x,y):
                if(1-tag*np.dot(self.w,pic) > 0):
                    self.w=(1-self.Pi*self.lamda)*self.w+pic*self.Pi*tag
                else:
                    self.w=(1-self.Pi*self.lamda)*self.w

    def predict(self,test_x,test_y):
        counter=0
        for i,ex in enumerate(test_x):
            if(sign(np.dot(self.w,ex)) == test_y[i]):
                counter+=1
        print("accuracy is: "+str((float(counter)/len(test_y))*100)+"%")



    #predicts value for a single example
    def singlePredict(self,ex):
        return sign(np.dot(self.w,ex))
        




# @@@@@ main @@@@ :

#getting test information
testing_y=np.loadtxt("y_test.txt")
testing_x=np.loadtxt("x_test.txt")




#remove this for one vs all:

#creating matrixes for the diffrent models
oneVsAllMatrix=[[1,-1,-1,-1],[-1,1,-1,-1],[-1,-1,1,-1],[-1,-1,-1,1]]

#creating svms One vs All
svms=[]
svms.append(SVM(fi,myLamda,epoches))
svms.append(SVM(fi,myLamda,epoches))
svms.append(SVM(fi,myLamda,epoches))
svms.append(SVM(fi,myLamda,epoches))



#running and testing
for i in range(len(svms)):
    svms[i].execute(x,change(float(i),y))
    #svms[i].predict(testing_x,change(float(i),testing_y))



hammingPred(svms,testing_x,testing_y,oneVsAllMatrix)
lossPred(svms,testing_x,testing_y,oneVsAllMatrix)








#creating svms All Pairs
svms=[]
svms.append(SVM(fi,myLamda,epoches))
svms.append(SVM(fi,myLamda,epoches))
svms.append(SVM(fi,myLamda,epoches))
svms.append(SVM(fi,myLamda,epoches))
svms.append(SVM(fi,myLamda,epoches))
svms.append(SVM(fi,myLamda,epoches))



allPairsMatrix=[[1,1,1,0,0,0],[-1,0,0,1,1,0],[0,-1,0,-1,0,1],[0,0,-1,0,-1,-1]]




listOfData=[]
listOfData.append(changeAllPairs(0.0,1.0,y,x))
listOfData.append(changeAllPairs(0.0,2.0,y,x))
listOfData.append(changeAllPairs(0.0,3.0,y,x))
listOfData.append(changeAllPairs(1.0,2.0,y,x))
listOfData.append(changeAllPairs(1.0,3.0,y,x))
listOfData.append(changeAllPairs(2.0,3.0,y,x))




#training
for i in range(6):
    svms[i].execute(listOfData[i][0],listOfData[i][1])


hammingPred(svms,testing_x,testing_y,allPairsMatrix)
lossPred(svms,testing_x,testing_y,allPairsMatrix)










#Random Matrix:
matrix=[]
xlen=20 #random.randint(4,20)


#class 0
indexX = [random.randint(-1,1) for i in range(xlen)]
matrix.append(indexX)

#class 1
indexX = [random.randint(-1,1) for i in range(xlen)]
matrix.append(indexX)

#class 2
indexX = [random.randint(-1,1) for i in range(xlen)]
matrix.append(indexX)

#class 3
indexX = [random.randint(-1,1) for i in range(xlen)]
matrix.append(indexX)



svms=[]
for i in range(xlen):
    svms.append(SVM(fi,myLamda,epoches))
    labels=[]
    for line in range(4):
        labels.append(matrix[line][i])

    svms[i].execute(x,changeRandom(y,labels[0],labels[1],labels[2],labels[3]))



hammingPred(svms,testing_x,testing_y,matrix)
lossPred(svms,testing_x,testing_y,matrix)
