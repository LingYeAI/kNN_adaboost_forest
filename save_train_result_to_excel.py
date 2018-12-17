# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 11:25:29 2018

@author: Shaofeng Zou
"""

import xlwt
from tempfile import TemporaryFile
import numpy as np

def saveTrainResult():
    trainResult = np.load('trainResult.npy')
    book = xlwt.Workbook()
    sheet1 = book.add_sheet('sheet1',cell_overwrite_ok=True)
    
    supersecretdata = trainResult
    sheet1.write(0,0,'trainingSet')
    sheet1.write(0,1,'k')
    sheet1.write(0,2,'accurcy')
    sheet1.write(0,3,'timeCost')
    for i,e in enumerate(supersecretdata):
        for j,d in enumerate(e):
            sheet1.write(i+1,j,d)
    
    name = "trainResult.xls"
    book.save(name)
    book.save(TemporaryFile())