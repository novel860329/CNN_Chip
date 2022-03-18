# -*- coding: utf-8 -*-
"""
Created on Fri May 31 08:26:14 2019

@author: user
"""
import csv
import numpy as np
import cv2 as cv

#local
print("Local圖片生產中.....",end="")
img = np.zeros((100, 100, 3), np.uint8)

for i in range(1,61):
    file_name = 'local/' + str(i) + '.csv'
    with open(file_name, newline='') as f:
        f = csv.reader(f)
        fieldnames = next(f)
        for row in f:      
            point = (int(row[0]), int(row[1]))
            if row[2] == 'P':
                cv.circle(img, point, 1, (0, 0, 255), -1)
            
    with open(file_name, newline='') as f:
        f = csv.reader(f)
        fieldnames = next(f)
        for row in f:      
            point = (int(row[0]), int(row[1]))
            if row[2] == 'F':
                cv.circle(img, point, 1, (255, 255, 255), -1)
    save_img = 'local/' + str(i) + '.jpg'
    cv.imwrite(save_img, img)
print("完成")

#none
print("None圖片生產中.....",end="")
img = np.zeros((100, 100, 3), np.uint8)

for i in range(1,295):
    file_name = 'none/' + str(i) + '.csv'
    with open(file_name, newline='') as f:
        f = csv.reader(f)
        fieldnames = next(f)
        for row in f:      
            point = (int(row[0]), int(row[1]))
            if row[2] == 'P':
                cv.circle(img, point, 1, (0, 0, 255), -1)
            
    with open(file_name, newline='') as f:
        f = csv.reader(f)
        fieldnames = next(f)
        for row in f:      
            point = (int(row[0]), int(row[1]))
            if row[2] == 'F':
                cv.circle(img, point, 1, (255, 255, 255), -1)
    save_img = 'none/' + str(i) + '.jpg'
    cv.imwrite(save_img, img)
print("完成")
    
#scratch
print("Scratch圖片生產中.....",end="")
img = np.zeros((100, 100, 3), np.uint8)

for i in range(1,53):
    file_name = 'scratch/' + str(i) + '.csv'
    with open(file_name, newline='') as f:
        f = csv.reader(f)
        fieldnames = next(f)
        for row in f:      
            point = (int(row[0]), int(row[1]))
            if row[2] == 'P':
                cv.circle(img, point, 1, (0, 0, 255), -1)
            
    with open(file_name, newline='') as f:
        f = csv.reader(f)
        fieldnames = next(f)
        for row in f:      
            point = (int(row[0]), int(row[1]))
            if row[2] == 'F':
                cv.circle(img, point, 1, (255, 255, 255), -1)
    save_img = 'scratch/' + str(i) + '.jpg'
    cv.imwrite(save_img, img)
print("完成")