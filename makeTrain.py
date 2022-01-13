# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:05:39 2019

@author: lisp
"""

import os

# make train.txt
path = 'data/custom/images/'

file_list = os.listdir(path)
file_list = [ file for file in file_list if file.endswith('.txt')]
file_list.sort()

path = 'data/custom/images/'
train_txt = open('train.txt', 'w')

for file in file_list :
    train_txt.write(path + "{}\n".format(file.replace(".txt", ".png")))
    
train_txt.close()    


# make valid.txt
path = 'data/custom/validation/'

file_list = os.listdir(path)
file_list = [ file for file in file_list if file.endswith('.txt')]
file_list.sort()

valid_txt = open('data/custom/valid.txt', 'w')

for file in file_list :
    valid_txt.write(path + "{}\n".format(file.replace(".txt", ".png")))
    
valid_txt.close()