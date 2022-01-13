# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:27:32 2019

@author: lisp
"""

import cv2
import os
import imutils
import numpy as np

def save_rotation(path, filename) :
    image = cv2.imread(path+filename, cv2.IMREAD_COLOR)
    
    #90 degree clockwise rotation
    image = imutils.rotate_bound(image, 90)
    cv2.imwrite(path+'90_{}'.format(filename), image)
    
    #rotate one more(180 degree)
    image = imutils.rotate_bound(image, 90)
    cv2.imwrite(path+'180_{}'.format(filename), image)

    #rotate one more(270 degree = 90 degree towards counter-clockwise)
    image = imutils.rotate_bound(image, 90)
    cv2.imwrite(path+'270_{}'.format(filename), image) 


def save_rotated_annotation(txt_path, filename) :
    ori_txt = open(txt_path+filename, 'r')
    txt_90 = open(txt_path+'90_{}'.format(filename), 'w')
    txt_180 = open(txt_path+'180_{}'.format(filename), 'w')
    txt_270 = open(txt_path+'270_{}'.format(filename), 'w') 
    
    for line in ori_txt :        
        line = line.split()
        dist = line[0]        
        x = float(line[1])
        y = float(line[2])
        w = float(line[3])
        h = float(line[4])
        txt_90.write('{} {} {} {} {}\n'.format(dist, 1-y, x, h, w))
        txt_180.write('{} {} {} {} {}\n'.format(dist, 1-x, 1-y, w, h))
        txt_270.write('{} {} {} {} {}\n'.format(dist, y, 1-x, h, w))

        
    ori_txt.close()
    txt_90.close()
    txt_180.close()
    txt_270.close()


reg_path = './reg_yolo_annotation/'
file_list = os.listdir(reg_path)
file_list = [ file for file in file_list if file.endswith('.txt')]
file_list.sort()

#
#print('calculating max value...')
#max_value = 0.
#for file in file_list :
#    a = open(reg_path+file, 'r')
#    
#    for line in a :
#        line = line.split()
#        dist = float(line[0])
#        max_value = np.maximum(max_value, dist)
#    a.close()
#print("max value is {}".format(max_value))
#


#
#print('saving rotated annotations...')
#for file in file_list :
#    save_rotated_annotation(reg_path, file)
#print('completed saving rotated annotation')
#


#
#print("reg to class converting...")
#clas_path = './clas_yolo_annotation/'
#for file in file_list :    
#    reg = open(reg_path + file, 'r')
#    clas = open(clas_path + file, 'w')
#    lines = reg.readlines()
#    for j in range(len(lines)) :
#        line = lines[j]
#        line = line.split()
#        line[0] = round(float(line[0]))
#        clas.write("{} {} {} {} {}\n".format(line[0], line[1], line[2], line[3], line[4]))        
#    clas.close()
#    reg.close()       
#print('completed converting reg to class')
#    
    
    
print('saving rotated images...')
path = './image/'
file_list = os.listdir(path)
img_files = [file for file in file_list if file.endswith('.png')]
img_files.sort()
for image in img_files :
    save_rotation(path, image)
print('completed saving rotated images')
    
