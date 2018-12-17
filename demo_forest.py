# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 16:59:43 2018

@author:
"""
from math import log
import numpy

def readFile(file_address,featNum,mode):
    """
    input:address of txt file , axis of feature , read mode rgb/gray
    output:list of img,number of img,label list
    description:read train or test file
    """
    """
    read all img files
    """
    img=[[]]
    img_num=-1  #number of imgs
    ID=[]
    file=open(file_address) #open train file
    lines=file.readlines() #read all data
    for i,line in enumerate(lines):  
        img_num+=1
        split_img=line.split() #split a single line by space
        img[i]=list(map(int,split_img[1:])) #trun str into int  
        ID.append(split_img[0])
        img.append([])
#        if i==999: break
    del img[img_num+1] #delete the extra line
    """
    treat the read mode
    """
    if mode=='gray':
        if featNum>64:
            return [],0,[],[]
        grayImg=[]
        for i in range(img_num+1):
            grayImg.append([])
            grayImg[i].append(img[i][0])
            for j in range(64):
                grayImg[i].append(int(img[i][j*3+1]*0.299+img[i][j*3+2]*0.587+img[i][j*3+3]*0.114))
        for i in range(img_num+1):
            grayImg[i]=grayImg[i][0:featNum+1]
        labels=list(range(1,featNum+1))
        return grayImg,img_num,labels,ID
    
    for i in range(img_num+1):
        img[i]=img[i][0:featNum+1]
    labels=list(range(1,featNum+1))
    return img,img_num,labels,ID

def calcEnt(dataSet):
    """
    input:a list of a dataset
    output:the shannon entropy of a dataset
    description:the number of class if fixed to 4
    """
    numEntries = len(dataSet) #这里应该是数据组的个数
    #print('len'+'='+str(len(dataSet)))
    labelCounts={}
    for featVec in dataSet:
        currentLabel = featVec[0]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt=0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet,axis,value,flag):
    """
    input:dataset , axis of feature (1,2,3,... and 0 is for tag), threshold , flag:'<' or'>'
    output:subdataset
    description:remove the member bigger than this value
    """  
    retDataSet=[]
    if flag=='<' or flag==0:
        for featVec in dataSet:
            if featVec[axis]<=value:
                reduceFeatVec=featVec[:axis]
                reduceFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reduceFeatVec)
    elif flag=='>'or flag==1:
        for featVec in dataSet:
            if featVec[axis]>value:
                reduceFeatVec=featVec[:axis]
                reduceFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reduceFeatVec)               
    return retDataSet

def calcInfoGain(dataSet,axis,value,baseEntropy):   
    newEntropy = 0.0 
    
    subDataSet = splitDataSet(dataSet,axis,value,'<')
    prob = len(subDataSet)/(len(dataSet))    
    newEntropy += prob * calcEnt(subDataSet)
    
    subDataSet = splitDataSet(dataSet,axis,value,'>')
    prob = len(subDataSet)/(len(dataSet))    
    newEntropy += prob * calcEnt(subDataSet)
    
    return baseEntropy-newEntropy
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])#real feature num +1,used for below
    #print('numfeat'+'='+str(numFeatures-1))
    baseEntropy = calcEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestThres = -1
    for i in range(1,numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = list(set(featList)) 
        uniqueVals.remove(max(uniqueVals))#remove the largest one because you will not split them into 2 sets
#        newEntropy = 0.0
#        splitInfo = 0.0
        for value in uniqueVals:
            infoGain=calcInfoGain(dataSet,i,value,baseEntropy)
#        infoGain = baseEntropy - newEntropy
#        if (splitInfo == 0): # fix the overflow bug
#            continue
#        infoGainRatio = infoGain / splitInfo
            if (infoGain >= bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i
                bestThres = value
    if bestFeature==-1:
        print(dataSet)
    return bestFeature,bestThres

def majorityCnt(classList):
    """
    Input: list of classification categories
    Output: Classification of child nodes
    Description: The dataset has processed all the properties, but the class label is still not unique.
                 The majority decision method is used to determine the classification of the child node.
    """
    classCount = [0,0,0,0]
    for vote in classList:
        classCount[int(vote/90)] += 1
    maxcnt=classCount.index(max(classCount))
    return maxcnt*90  

def createTree(dataSet, labels):
    """
    Input: data set, feature tag
    Output: Decision Tree
    Description: Recursively build a decision tree, using the above functions
    """
    #print('labellen'+str(len(labels)))
    classList = [example[0] for example in dataSet]
#    print(classList)
    if classList.count(classList[0]) == len(classList):
        # 类别完全相同，停止划分
        return classList[0]
    if len(dataSet[0])>=2:
        featList=[dataset[1:] for dataset in dataSet]
        if(featList.count(featList[0])==len(featList)):
            return majorityCnt(classList)
    if len(dataSet[0]) == 1 :  
        # 遍历完所有特征时返回出现次数最多的;或者所有特征都相同了，无法继续划分
        return majorityCnt(classList)
    bestFeat , bestThres = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat-1]
    myTree = {bestFeatLabel:{}}
    #print('feature'+str(bestFeat))    
    
    del(labels[bestFeat-1])
    # 得到列表包括节点所有的属性值
    #featValues = [example[bestFeatLabel] for example in dataset]
    #uniqueVals = list(set(featValues))
    for i in ['<','>']:        
        subLabels = labels[:]
        myTree[bestFeatLabel][i+str(bestThres)] = createTree(splitDataSet(dataSet, bestFeat, bestThres,i), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    """
    Input: decision tree, classification label, test data
    Output: Decision Results
    Description: Run the decision tree
    """
    firstNum = list(inputTree.keys())
    secondDict = inputTree[firstNum[0]]
    #featIndex = featLabels.index(firstStr)
    allKeys=list(secondDict.keys())
    leftKey=allKeys[0]
    rightKey=allKeys[1]
    thresNum=int(leftKey[1:])
    if testVec[firstNum[0]] <= thresNum:
        if type(secondDict[leftKey]).__name__ == 'dict':
            classLabel = classify(secondDict[leftKey], featLabels, testVec)
        else:
            classLabel = secondDict[leftKey]
    else:
        if type(secondDict[rightKey]).__name__ == 'dict':
            classLabel = classify(secondDict[rightKey], featLabels, testVec)
        else:
            classLabel = secondDict[rightKey]
    return classLabel

def classifyAll(inputTree, testDataSet):
    """
    Input: decision tree, classification label, test data
    Output: Decision Results
    Description: Run the decision tree...
    """
    featLabels=list(range(1,len(testDataSet[0])))
    classLabelPredict = []
    correctNum=0
    for testVec in testDataSet:
        classLabelPredict.append(classify(inputTree, featLabels, testVec))
    classList = [example[0] for example in testDataSet]
    for i in range(len(classList)):
        if classList[i]==classLabelPredict[i]:
            correctNum+=1    
    return correctNum/len(classList),classLabelPredict

def voteClassify(randomForest,testDataSet):
    results=[]
    correctNum=0
    for decisionTree in randomForest:
        _,classLabelPredict=classifyAll(decisionTree,testDataSet)
        results.append(classLabelPredict)
    predictList=[]
    for i in range(len(classLabelPredict)):
        tempList=[0,0,0,0]
        for singleResultList in results:
            tempList[int(singleResultList[i]/90)]+=1
        predictList.append(90*tempList.index(max(tempList)))
    classList = [example[0] for example in testDataSet]
    for i in range(len(classList)):
        if classList[i]==predictList[i]:
            correctNum+=1
    return correctNum/len(classList),predictList 

def storeTree(inputForest, filename,featNum,mode):
    """
    Input: decision tree, save file path,number of features,mode
    Output:
    Description: Save the decision tree to the file
    """
    f=open(filename,'w')
    f.writelines(str(featNum)+' '+mode+'\n')
    for tree in inputForest:
        f.writelines(str(tree)+'\n')
    f.close()
    
def readTree(filename):
    """
    input:filename
    output:forest,feature number,mode
    dicription:
    """
    forest=[]
    f=open(filename)
    lines=f.readlines()
    parameters=lines[0].split()
    featNum=int(parameters[0])
    mode=parameters[1]
    del(lines[0])
    for line in lines:
        forest.append(eval(line))
    f.close()
    return forest,featNum,mode    

def model_forest_train(train_file='train-data.txt',model_file='forest-model.txt',featNum=32,mode='rgb',treeNum=50):
    allTrainImg,num,labels,_=readFile(train_file,featNum,mode)
    print('Read %s for testing successfully!!\n'%(train_file))
    i=0
    randomForest=[]
    while i<treeNum:
        trainSet=[]
        selectList=numpy.random.randint(0,len(allTrainImg)-1,500)
        for j in selectList:
            trainSet.append(allTrainImg[j])
        labels=list(range(1,featNum+1))
        newTree=createTree(trainSet,labels)
        randomForest.append(newTree)
        if i<1:
            print(str(i+1)+'/'+str(treeNum)+' tree has been trained')
        elif i>=1:
            print(str(i+1)+'/'+str(treeNum)+' trees have been trained')
        i+=1
    storeTree(randomForest,model_file,featNum,mode)
    print('All trees have been stored')
    
def model_forest_test(test_file='test-data.txt',model_file='knn-model.txt'):
    reloadForest,featNum,mode=readTree(model_file) 
    testImg,num,label,ID=readFile(test_file,featNum,mode)
    results,predictList=voteClassify(reloadForest,testImg)
    print('The accuracy is '+str(results))
    outputFile=open('output.txt','w')
    for i in range(len(ID)):
        outputFile.write(ID[i])
        outputFile.write(' ')
        outputFile.write(str(predictList[i]))
        outputFile.write('\n')
    print('Output file has been stored.')    
    
    
    