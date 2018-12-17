# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:00:39 2018
This is the main file.
@author:
"""

import sys
import demo_knn
import demo_forest
import demo_adaboost

if sys.argv[1]=='train':
    trainFile=sys.argv[2]
    modelFile=sys.argv[3]
    if sys.argv[4]=='nearest':
        print('knn')
        demo_knn.module_knn_train(trainFile,modelFile)
    elif sys.argv[4]=='adaboost':
        demo_adaboost.model_adaboost_train(trainFile,modelFile)
    elif sys.argv[4]=='forest':
        demo_forest.model_forest_train(trainFile,modelFile)
    elif sys.argv[4]=='best':
        print("We will use a pretrained model which has been trained for a long time.")
elif sys.argv[1]=='test':
    testFile=sys.argv[2]
    modelFile=sys.argv[3]
    if sys.argv[4]=='nearest':
        demo_knn.model_knn_test(testFile,modelFile)
    elif sys.argv[4]=='adaboost':
        demo_adaboost.model_adaboost_test(testFile,modelFile)
    elif sys.argv[4]=='forest':
        demo_forest.model_forest_test(testFile,modelFile)
    elif sys.argv[4]=='best':
        demo_forest.model_forest_test(testFile,'pretrainedModel.txt')