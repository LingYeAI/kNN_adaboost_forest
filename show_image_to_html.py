# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 16:34:07 2018

@author: Shaofeng Zou
"""
import numpy as np
import cv2

def flip_90(img):
    w = img.shape[1]
    h = img.shape[0]
    img2= np.zeros([img.shape[1], img.shape[0], 3], np.uint8)
    for i in range(w):
        for j in range(h):
            img2[i,j] = img[h-1-j,i]         
    return img2

def flip_180(img):
    w = img.shape[1]
    h = img.shape[0]
    img2= np.zeros([img.shape[0], img.shape[1], 3], np.uint8)
    for i in range(h):
        for j in range(w):
            img2[i,j] = img[h-1-i,w-1-j]         
    return img2

def flip_270(img):
    w = img.shape[1]
    h = img.shape[0]
    img2= np.zeros([img.shape[1], img.shape[0], 3], np.uint8)
    for i in range(w):
        for j in range(h):
            img2[i,j] = img[j,i]         
    return img2
def flip_0(img):
    w = img.shape[1]
    h = img.shape[0]
    img2= np.zeros([img.shape[0], img.shape[1], 3], np.uint8)
    for i in range(h):
        for j in range(w):
            img2[i,j] = img[i,j]         
    return img2

def show_image_on_html(htmlFile,predictSet):
    index = open(htmlFile,'w')
    index.write("<html><body><table><tr>")
    index.write("<th>img_id</th><th>\t img_origin</th><th>img_reality</th><th>img_guess</th><th>  \t guess_result</th></tr>")
    
     
    for vector in predictSet:    
        imgTestId = 'a5-photo-data/' + vector[0]
        imgTestOrientation = vector[1]
        guessOrientation = vector[2]
        
#        print(imgTestId)
#        print(imgTestOrientation)
#        print(guessOrientation)
        img = cv2.imread(imgTestId)
        
        if imgTestOrientation == '90':
            img_reality = flip_90(img)
        elif imgTestOrientation == '180':
            img_reality = flip_180(img)
        elif imgTestOrientation == '270':
            img_reality = flip_270(img)
        else:
            img_reality = flip_0(img)
            
        if guessOrientation == '90':
            img_guess = flip_90(img)
        elif guessOrientation == '180':
            img_guess = flip_180(img)
        elif guessOrientation == '270':
            img_guess = flip_270(img)
        else:
            img_guess = flip_0(img)
            
        imgId = imgTestId.split('/')[2].split('.')[0]
        
        img_reality_address = 'a5-photo-data/temp/'+ 'reality'+imgId+'.jpg'
        img_guess_address = 'a5-photo-data/temp/'+ 'guess'+imgId+'.jpg'
        cv2.imwrite(img_reality_address,img_reality)
        cv2.imwrite(img_guess_address,img_guess)
    
        
        index.write("<td>%s</td>"%(imgTestId))
        index.write("<td><img src= %s></td>"%(imgTestId))
        index.write("<td><img src= %s></td>"%(img_reality_address))
        index.write("<td><img src= %s></td>"%(img_guess_address))
        
        if imgTestOrientation == guessOrientation:
            index.write("<td>%s</td>"%('Right'))
        else:
            index.write("<td>%s</td>"%('Wrong'))
        index.write("<td>%s</td>"%(imgTestOrientation))
            
        index.write("</tr>")  
        
    index.close()

"""
@funciton: 根据输出html文件名及预测正确和错误的训练集生成两个html文件和temp文件
@input:      htmlTrueFile = 'test_true_result.html'
             htmlFalseFile = 'test_false_result.html'
             predictTrueSet
             predictFalseSet
@output: 'test_true_result.html'  'test_false_result.html'
"""
def show_result_on_html(predictTrueSet,htmlTrueFile,predictFalseSet,htmlFalseFile):

    show_image_on_html(htmlTrueFile,predictTrueSet)
    show_image_on_html(htmlFalseFile,predictFalseSet)