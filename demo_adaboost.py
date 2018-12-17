# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:29:39 2018

@author:
"""

import numpy as np
import demo_forest
import demo_knn
import show_image_to_html
#def loadSimpData():
#    datMat = np.mat([[ 1. ,  2.1],
#        [ 2. ,  1.1],
#        [ 1.3,  1. ],
#        [ 1. ,  1. ],
#        [ 2. ,  1. ]])
#    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
#    return datMat,classLabels

  
 

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = np.ones((np.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    dataMatrix = np.mat(dataArr); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = np.mat(np.zeros((m,1)))
    minError = np.inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = np.mat(np.ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

 
def adaBoostTrainDS(dataArr,classLabels,numIt=400):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m,1))/m)   #init D to all equal
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
#        print "D:",D.T
        alpha = float(0.5*np.log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
#        print "classEst: ",classEst.T
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = np.multiply(D,np.exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
#        print "aggClassEst: ",aggClassEst.T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    dataMatrix = np.mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
#        print(aggClassEst)
    return aggClassEst
#    return np.sign(aggClassEst)
 
def model_adaboost_train(train_file='train-data.txt',model_file='forest-model.txt'):
    featNum = 64*3
    mode = 'rgb'     
    #获得数据集
    img,img_num,splitLabels,_ = demo_forest.readFile(train_file,featNum,mode)
    dataArr = []
    classLabels = []
    for example in img:
        dataArr.append(example[1:])
        if example[0] ==0:
            label = 1
        else:
            label = -1
        classLabels.append(label)
        
    weakClassArr,aggClassEst = adaBoostTrainDS(dataArr,classLabels)
    f=open(model_file,'w')
    for classifier in weakClassArr:
        f.writelines(str(classifier)+'\n')
    f.close()
    print('Adaboost classifier has been stored.')
    return weakClassArr,aggClassEst



def model_adaboost_test(test_file='test-data.txt',model_file='knn-model.txt'):
    print('Testing,please wait...')
    featNum = 64*3
    mode = 'rgb'
    img_test,img_num_test,splitLabels_test,ID = demo_forest.readFile(test_file,featNum,mode)
    dataArr_test = []
    classLabels_test = []
    for example in img_test:
        dataArr_test.append(example[1:])
        if example[0] ==0:
            label = 1
        else:
            label = -1
        classLabels_test.append(example[0])
    #read model from model_file    
    weakClassArr=[]    
    f=open(model_file)
    lines=f.readlines()  
    for line in lines:
        weakClassArr.append(eval(line))
    f.close()
    
    index = 0
    right_cnt = 0
    predictList=[] 
     
    for data in dataArr_test:
        r= []
        g= [] 
        b= []
        for i in range(len(data)):
            if i%3 == 0:
                r.append(data[i])
            elif i%3 == 1:
                g.append(data[i])
            elif i%3 == 2:
                b.append(data[i])
        r = np.reshape(np.array(r),(8,8))
        g = np.reshape(np.array(g),(8,8))
        b = np.reshape(np.array(b),(8,8))
        rotation_predicts = []
        for i in range(0,4):
            rot_r = np.rot90(r,i)
            rot_g = np.rot90(g,i)
            rot_b = np.rot90(b,i)
            rot_r_line = np.reshape(rot_r,(1,64))
            rot_g_line = np.reshape(rot_g,(1,64))
            rot_b_line = np.reshape(rot_b,(1,64))
     
            data_test =[]
            for j in range(len(rot_g_line[0])):
                data_test.extend([rot_r_line[0][j],rot_g_line[0][j],rot_b_line[0][j]])
            rotation_predict = adaClassify(data_test,weakClassArr)
            rotation_predicts.append(rotation_predict)
    #        print(rotation_predict)
        predict = rotation_predicts.index(max(rotation_predicts)) *90
        predictList.append(predict)
#        print('predict',predict)
#        print('true rotation:',img_test[index][0])
        if(predict == img_test[index][0]):
            right_cnt = right_cnt + 1
        index = index + 1
    print('The accuracy is ',right_cnt/len(img_test))
    
    outputFile=open('output.txt','w')   
    for i in range(len(ID)):
        outputFile.write(ID[i])
        outputFile.write(' ')
        outputFile.write(str(predictList[i]))
        outputFile.write('\n')
    print('Output file has been stored.')        
    