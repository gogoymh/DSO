# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:36:41 2019

@author: lisp
"""


import os
import sys
import numpy as np
import cv2
from PIL import Image
from mrcnn import utils
import mrcnn.model as modellib

def get_min_distance(depth, masks, boxes) :
    # param(depth) : [H, W] height H, and width W of depth map image     
    # param(masks) : [H, W, count] count is the number of detected objects
    # param(boxes) : [count, (y1,x1, y2,x2)] 
    # return : minimum distance [count, distance]
    distance = []
    
    for i in range(masks.shape[2]) :        
        box = boxes[i]
        
        # find minimum distance where depth is not '-1'
        depth_crop = depth[box[0]:box[2], box[1]:box[3]]
        mask_crop = masks[box[0]:box[2], box[1]:box[3], i]
                
        b_depth = depth_crop > -1
        b_depth = np.logical_and(b_depth, mask_crop)
        result = depth_crop[b_depth]
        
        #print(i, box)
        #print(result.shape)        
        
        if (result.shape[0] == 0) : # depth not exists
            distance.append(-1)
        else :
            distance.append(np.min(result))
        

        
    distance = np.array(distance)
    return distance
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



#############################################################################
#                   from here you can customize code                        #
#############################################################################
val_path = './depth_selection/val_selection_cropped/'
val_img_path = val_path+'image/'
val_raw_path = val_path+'velodyne_raw/'
file_list = os.listdir(val_img_path)
img_files = [file for file in file_list if file.endswith('.png')]
file_list = os.listdir(val_raw_path)
raw_files = [file for file in file_list if file.endswith('.png')]
img_files.sort()
raw_files.sort()

# Run detection
j=1
for filename in img_files:
    print('{}-th image detection : )'.format(j))
    
    img = cv2.imread(val_img_path+filename, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    raw = np.array(Image.open(val_raw_path+filename), dtype=int)
    depth = raw.astype(np.float) / 256. # calculate depth as metric of meters
    depth[raw == 0] = -1. # if data is not detected, set '-1'
        
    results = model.detect([img], verbose=0)
    r = results[0]
    boxes = r['rois']
    masks = r['masks']
              
    # get minimum distance
    min_dist = get_min_distance(depth, masks, boxes)
    #print(min_dist)    
    # Generate ground-truth as form of [count, (distance, [x1, y1, x2, y2])] 
    file = open('{}'.format(filename.replace('.png', '.txt')), 'w')
    file.write('{}\n'.format(len(min_dist[min_dist>-1])))
    for i, box in enumerate(boxes) :
        if (min_dist[i] > -1) :
            y1, x1, y2, x2 = box
            yolo_box = [((x1+x2) / 2) / width, ((y1+y2) / 2) / height, (x2-x1) / width, (y2-y1) / height]    
            file.write('{} {} {} {} {}\n'.format(min_dist[i], yolo_box[0], yolo_box[1], yolo_box[2], yolo_box[3]))
    file.close()
        
    j=j+1     
    
    #print(boxes)   # for debugging 
    #print(masks.shape)
    #print(depth.shape)
    #print(np.min(depth))


#"""
#    # Make text files and Make image files using text information
#    for box in boxes :
#        visualize.draw_box(img, box, color)
#    cv2.imwrite('{}.png'.format(j), img)
#    file = open('{}.txt'.format(j), 'w')
#    file.write('{}\n'.format(boxes.shape[0]))
#    file.write('{}'.format(boxes))
#    file.close()
#"""
#
#
#
#
#
#
## Load a random image from the images folder
##file_names = next(os.walk(IMAGE_DIR))[2]
##image = cv2.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
#""" 
## read images, raws and re-draw images with masks.
## make txt annotation files with format(detection number and [det_num, (offset of the bounding boxes)])
#
#img_path = '../images/'
#raw_path = '../raws/'
#file_list = os.listdir(img_path)
#img_files = [file for file in file_list if file.endswith('.png')]
#file_list = os.listdir(raw_path)
#raw_files = [file for file in file_list if file.endswith('.png')]
#img_files.sort()
#raw_files.sort()
#                
#color = [250, 250, 250]
#j=0
## Run detection
#for image in img_files:
#    print('{}-th image detection : )'.format(j))
#    
#    img = cv2.imread(img_path+image, cv2.IMREAD_COLOR)
#    raw = np.array(Image.open(raw_path+image), dtype=int)
#    depth = raw.astype(np.float) / 256. # calculate depth as metric of meters
#    depth[raw == 0] = -1. # if data is not detected, set '-1'
#        
#    results = model.detect([img], verbose=0)
#    r = results[0]
#    boxes = r['rois']
#    masks = r['masks']
#    #print(masks.shape)
#    #class_ids = r['class_ids']  
#    '''
#    # save data as masked image
#    for i in range(masks.shape[2]) :
#        mask = masks[:,:,i]
#        img[mask] = color
#    cv2.imwrite('mask_{}.png'.format(j), img)
#    '''
#              
#    # get minimum distance
#    min_dist = get_min_distance(depth, masks, boxes)
#    #print(min_dist)    
#    # Generate ground-truth as form of [count, (distance, [x1, y1, x2, y2])] 
#    file = open('{}'.format(image.replace('.png', '.txt')), 'w')
#    file.write('{}\n'.format(len(min_dist[min_dist>-1])))
#    for i, box in enumerate(boxes) :
#        if (min_dist[i] > -1) :
#            file.write('[{}, {}]\n'.format(min_dist[i], box))
#    file.close()
#        
#    j=j+1     
#    
#    #print(boxes)   # for debugging 
#    #print(masks.shape)
#    #print(depth.shape)
#    #print(np.min(depth))
#"""
#    
## read texts, find directory in texts, and then read the images
## raw + object data mapping path
##txt_path = '../../kitti/generated_annotation/'
##raw_path = '../../kitti/data_depth_velodyne/train/'
##img_path = '../../kitti/data_object_image_3/training/image_3/'
##mapping_file = 'train_mapping.txt'
##rand_file = 'train_rand.txt'
#
#
## test_completion path
#txt_path = '../../kitti/depth_selection/annotation_completion/'
#raw_path = '../../kitti/depth_selection/test_depth_completion_anonymous/velodyne_raw/'
#img_path = '../../kitti/depth_selection/test_depth_completion_anonymous/image/'
#
## read rand file and mapping file(raw + object data mapping MODE)
##rand_txt = open(rand_file, 'r')
##mapping_file = open(mapping_file, 'r')
##rand_arr = rand_txt.readline()
##rand_arr = [int(index) for index in rand_arr.split(',')]
##mapping_arr = mapping_file.readlines()
##rand_txt.close()
##mapping_file.close()
#
## make the annotation text files of image files
#file_list = os.listdir(img_path)
#img_files = [file for file in file_list if file.endswith('.png')]
#img_files.sort()
#count = 0 # how many files doesn't exist.
##for i, rand_idx in enumerate(rand_arr) : # (raw + object data mapping MODE)
#for i, image in enumerate(img_files) :
#    print('{}-th image detection : )'.format(i+1))
#    
#    # (raw + object data mapping MODE)
#    #mapping_str = mapping_arr[rand_idx - 1].split()    
#    #dir_path = mapping_str[1] + '/proj_depth/velodyne_raw/image_03/'
#    #raw_name = mapping_str[2] + '.png'
#    
#    
#    """
#    img = cv2.imread(img_path+img_files[i], cv2.IMREAD_COLOR)    
#    if (os.path.isfile(raw_path + dir_path + raw_name)) :
#        raw = np.array(Image.open(raw_path + dir_path + raw_name), dtype=int)
#    else : 
#        count = count + 1
#        continue
#    """
#    img = cv2.imread(img_path+image, cv2.IMREAD_COLOR)
#    raw = np.array(Image.open(raw_path + image), dtype = int)
#    depth = raw.astype(np.float) / 256. # calculate depth as metric of meters
#    depth[raw == 0] = -1. # if data is not detected, set '-1'
#    
#    results = model.detect([img], verbose=0)
#    r = results[0]
#    boxes = r['rois']
#    masks = r['masks']
#
#    # get minimum distance
#    min_dist = get_min_distance(depth, masks, boxes)
#    
#    # save data as masked image
#    for j in range(masks.shape[2]) :
#        mask = masks[:,:,j]
#        if (min_dist[j] > -1) :
#            img[mask] = [min_dist[j], min_dist[j], min_dist[j]]
#    
#        
#    #cv2.imwrite('mask/mask_{}'.format(img_files[i]), img)
#    cv2.imwrite(raw_path+'../../mask/mask_{}'.format(img_files[i]), img)
#    
#    
#    # Generate ground-truth as form of [count, (distance, [y1, x1, y2, x2])] 
#    file = open(txt_path+'{}'.format(img_files[i].replace('.png', '.txt')), 'w')
#    file.write('{}\n'.format(len(min_dist[min_dist>-1])))
#    for i, box in enumerate(boxes) :
#        if (min_dist[i] > -1) :
#            file.write('[{}, {}]\n'.format(min_dist[i], box))
#    file.close()
#    
#print(count)