# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:16:44 2018

@author:
"""


import numpy as np 
import time
import save_train_result_to_excel 
import show_image_to_html
#import matplotlib.pyplot as plt

"""
@funciton:读取训练集中的数据
@input:训练集地址
@output:图片ID,图片方向，图片的特征向量
"""
def readTrainData(filename):  
    
    fr = open(filename)
    arrayLines = fr.readlines()
    numberLines = len(arrayLines)
    imgIds = []
    imgLabels = []
    imgDataSet = np.zeros((numberLines,192),dtype =int)
     
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(' ')
        imgIds.append(listFromLine[0])
        imgLabels.append(listFromLine[1])
        imgDataSet[index,:] = listFromLine[2:]
        index = index + 1
        
    return imgIds,imgLabels,imgDataSet

""""
@funciton:kNN分类
@input:待分类图像特征向量，数据集，数据集的标签，k值
@output:分类结果
"""
def kNN_classify(targetImg, dataSet, labels, k):
    
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(targetImg, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    #sort the index
    sortedDistIndex = np.argsort(distances)     
    classCount={}          
    for i in range(k):
        voteLabel = labels[sortedDistIndex[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key = lambda classCount:classCount[1], reverse=True)
    
    return sortedClassCount[0][0]


""""
@funciton:根据给定的训练集和测试集,在当前的k值下判断测试集分类准确度
@input:训练集，测试集，k值
@output:分类准确度
"""
def testKNNAccurcy(imgTrainIds,imgTrainOrientations,imgTrainVectors,imgTestIds,imgTestOrientations,imgTestVectors,k):
    predictTrueNum = 0
    predictFalseNum = 0
    
    predictSet=[]
    predictTrueSet = []
    predictFalseSet = []
    outputLines = []
    for i in range(len(imgTestIds)):
        guessOrientation = kNN_classify(imgTestVectors[i], imgTrainVectors, imgTrainOrientations, k)
        outputLine = [imgTestIds[i],guessOrientation]
        outputLines.append(outputLine)
        if guessOrientation == imgTestOrientations[i]:
            predictTrueNum = predictTrueNum + 1
            predictTrueSet.append([imgTestIds[i],imgTestOrientations[i],guessOrientation])
        else:
            predictFalseNum = predictFalseNum +1    
            predictFalseSet.append([imgTestIds[i],imgTestOrientations[i],guessOrientation])
        predictSet.append([imgTestIds[i],imgTestOrientations[i],guessOrientation])
    accurcy = predictTrueNum/len(imgTestIds)
    return accurcy,predictSet,predictTrueSet,predictFalseSet

"""
@function:根据给定的训练集地址及k值，获得当前训练集下获得在此训练集下的分类准确度
@input:训练集，k值
@output:分类准确度
"""
def trainKNN(imgTrainIds,imgTrainOrientations,imgTrainVectors,k):   

    
    #把训练集拆分成两份训练集，一份验证集
    valDataSetSize = int(len(imgTrainIds)/3)
    accurcyAverage = 0
    for i in range(3):
        #图片ID 
        temp = imgTrainIds.copy()
        partImgValIds = temp[i*valDataSetSize:(i+1)*valDataSetSize]
        del temp[i*valDataSetSize:(i+1)*valDataSetSize]
        partImgTrainIds = temp
        #图片方向
        temp = imgTrainOrientations.copy()
        partImgValOrientations = temp[i*valDataSetSize:(i+1)*valDataSetSize]
        del temp[i*valDataSetSize:(i+1)*valDataSetSize]
        partImgTrainOrientations = temp
        #图片特征向量
        partImgValVectors = imgTrainVectors[i*valDataSetSize:(i+1)*valDataSetSize]    
        partImgTrainVectors = np.zeros((len(imgTrainVectors)-valDataSetSize,192),dtype =int)
        t = 0
        temp1 = partImgValVectors.tolist()
        temp2 = imgTrainVectors.tolist()
        for vector in temp2:
            if vector not in temp1:
                partImgTrainVectors[t,:] = vector
                t = t + 1
                
        accurcy,predictSet,predictTrueSet,predictFalseSet =  testKNNAccurcy(partImgTrainIds,partImgTrainOrientations,partImgTrainVectors,partImgValIds,partImgValOrientations,partImgValVectors,k)
        accurcyAverage = accurcyAverage + accurcy
    
    #获得在当前训练集下及k值下的准确度
    accurcyAverage = accurcyAverage/3  
    
    return accurcyAverage,predictTrueSet,predictFalseSet

"""
@function:生成model.txt文件，包括最佳k值和从训练集解析的数据
@input:最佳k值和从训练集解析的数据
@output:model.txt
"""
def generateTrainModelFile(modelFilename,bestK,dataIds,dataSet, labels):
    file = open(modelFilename, 'w' )
    file.write(str(bestK))
    file.write('\n')
    for i in range(len(dataSet)):
        file.write(dataIds[i])
        file.write(' ')
        file.write(labels[i])
        file.write(' ')
        for element in dataSet[i]:
            file.write(str(element))
            file.write(' ')
        file.write('\n')
            
  
 
"""
@function:生成在各大小不同训练集及不同k值下的训练结果
@input:训练集，需生成大小不同训练集的个数，k值范围
@output:训练结果包括bestK,maxAccurcy,trainResult[splitTrainSetSize,k,accurcy,timeCost] 
"""
def generateTrainReport(imgTrainIds,imgTrainOrientations,imgTrainVectors,times_train,k_range): 
        
    accurcyDict ={}
    trainResult = []
    trainSetSize = 10000
    maxAccurcy = 0
    bestK = 1
    #根据数据集大小及不同大小训练集的总数生成最小训练集的大小，能被100整除
    trainSetSize = len(imgTrainIds)
    splitTrainSetSize = int(np.floor(trainSetSize/(times_train*100))*100)
    print('splitTrainSetSize',splitTrainSetSize)
    
    for i in range(times_train):
        #不同大小的训练集
        trainIds = imgTrainIds[i*splitTrainSetSize:(i+1)*splitTrainSetSize]
        trainOrientations = imgTrainOrientations[i*splitTrainSetSize:(i+1)*splitTrainSetSize] 
        trainVectors = imgTrainVectors[i*splitTrainSetSize:(i+1)*splitTrainSetSize] 
        
        #根据训练集获得训练结果
        for k in range(1,k_range):
            t1 = time.time()        
            accurcy,predictTrueSet,predictFalseSet = trainKNN(trainIds,trainOrientations,trainVectors,k) 
            t2 = time.time()
            timeCost = round(t2-t1,2)
            accurcy = round(accurcy,3)
            accurcyDict[k] = accurcy
            trainResult.append([(i+1)*splitTrainSetSize,k,accurcy,timeCost])
            
            print('trainSetSize: %d , k: %d , accurcy: %.3f , time cost: %.2f\n'%((i+1)*splitTrainSetSize,k,accurcy,timeCost)) 
        
            sortedAccurcy = sorted(accurcyDict.items(), key = lambda accurcyDict:accurcyDict[1], reverse=True)  
            if maxAccurcy < sortedAccurcy[0][1]:
                maxAccurcy = sortedAccurcy[0][1]  
                
    np.save('trainResult.npy',trainResult)
    return bestK,maxAccurcy,trainResult
    

"""
@function:根据训练集,获得最佳k值，并输出model_file
@input: train_file='train-data.txt'
@output: model_file='knn-model.txt'
"""    
def module_knn_train(train_file='train-data.txt',model_file='knn-model.txt',k_range = 3,times_train = 1):
    #解析训练集
    imgTrainIds,imgTrainOrientations,imgTrainVectors = readTrainData(train_file)
    print('Read %s for testing successfully!!\n'%(train_file))
    
    #生成在各大小不同训练集及不同k值下的训练结果
    bestK,maxAccurcy,trainResult = generateTrainReport(imgTrainIds,imgTrainOrientations,imgTrainVectors,times_train,k_range)
    print('In the case of k_range = %d, times_train = %d, training process has done!!'%(k_range,times_train))
    print('Result: best k value = %d, max accurcy = %.3f\n'%(bestK,maxAccurcy))

    #将结果保存至excel文件成功
    save_train_result_to_excel.saveTrainResult()
    print('Generate trainResult.xls in training successfully!!\n')
    
    #生成参数文件
    generateTrainModelFile(model_file,bestK,imgTrainIds,imgTrainVectors,imgTrainOrientations)
    print('Generate %s in training successfully!!\n'%(model_file))







"""
@funciton:读取model_file中的文件
@input:model_file
@output:最佳k值,图片ID,图片方向，图片的特征向量
"""
def readTrainModelFile(model_file):
    fr = open(model_file)
    arrayLines = fr.readlines()
    numberLines = len(arrayLines) -1 
    imgIds = []
    imgLabels = []
    imgDataSet = np.zeros((numberLines,192),dtype =int)
    bestK = 1
    
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(' ')
        if len(listFromLine) == 1:
            bestK = int(listFromLine[0])
        else:
            imgIds.append(listFromLine[0])
            imgLabels.append(listFromLine[1])
            imgDataSet[index,:] = listFromLine[2:]
            index = index + 1
        
    return bestK,imgIds,imgLabels,imgDataSet

"""
@function:生成output.txt文件，格式： test/124567.jpg 180
@input:output_filename , predictSet[imgTestIds[i],imgTestOrientations[i],guessOrientation]
@output:output.txt
"""
def generateTestOutputFile(outputFilename,predictSet):
    file = open(outputFilename, 'w' )

    for predict in predictSet:
        file.write(predict[0])
        file.write(' ')
        file.write(predict[2])
        file.write('\n')
        

"""
@funciton:根据测试集、参数模型来计算knn在测试集的分类准确度
@input:test_file='test-data.txt' , model_file='knn_model.txt'
@output: accurcy ,
@output:最佳k值,图片ID,图片方向，图片的特征向量
"""
def model_knn_test(test_file='test-data.txt',model_file='knn-model.txt'):
    
    #解析模型
    bestK,imgTrainIds,imgTrainOrientations,imgTrainVectors = readTrainModelFile(model_file)
    print('Read knn-model.txt for testing successfully!!\n')
    
    #解析测试集
    imgTestIds,imgTestOrientations,imgTestVectors = readTrainData(test_file)
    print('Read test-data.txt for testing successfully!!\n')
    
    #获取测试集分类结果
    accurcy,predictSet,predictTrueSet,predictFalseSet = testKNNAccurcy(imgTrainIds,imgTrainOrientations,imgTrainVectors,imgTestIds,imgTestOrientations,imgTestVectors,bestK)
    print('Testing have finished')
    print('Testing accurcy: %.3f\n'%(accurcy))
    
    #输出output.txt文件
    outputFilename = 'output.txt'
    generateTestOutputFile(outputFilename,predictSet) 
    print('Generate output.txt for testing successfully!!\n')
    
    
    #在html上画出正确分类和错误分类的所有照片
    htmlTrueFile = 'test_true_result.html'
    htmlFalseFile = 'test_false_result.html'           
    show_image_to_html.show_result_on_html(predictTrueSet,htmlTrueFile,predictFalseSet,htmlFalseFile)
    print('Generate test_true_result.html and test_false_result.html for testing successfully!!\n')