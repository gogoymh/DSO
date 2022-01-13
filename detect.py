from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import colorsys

from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

def random_colors(N):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/depth.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

#    if opt.weights_path.endswith(".weights"):
#        # Load darknet weights
#        model.load_darknet_weights(opt.weights_path)
#    else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.checkpoint_model))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    detectionstimes = []
    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
#            print(detections.shape) # torch.Size([1, 10647, 6])
#            print(detections) # tensor([[[4.0680e+00, 7.9177e-01, 2.6245e+00, 2.5575e+01, 2.6826e-13, 6.2991e+00], ...)

            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres) # Output is list-type with only one element. Use it with 'detections[0]'
            if detections[0] is not None :
                print('after nms, detections[0].size : {}'.format(detections[0].size)) # torch.Size([10, 7])
#            print(detections[0]) # tensor([[1.8497e+02, 1.9971e+02, 1.9322e+02, 2.0634e+02, 9.9745e-01, 5.8307e-01, 5.8307e-01], ...)
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        detectionstimes.append(inference_time)
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    #cmap = plt.get_cmap("tab20b")
    #colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    detectionstimes = np.array(detectionstimes)
    print('mean time to detect a image : {}'.format(detectionstimes[1:].mean()))
    
    print("\nSaving images...")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

#        print("(%d) Image: '%s'" % (img_i, path))
        if detections is None :
            continue
        N=detections.shape[0]
        colors = random_colors(N)
        
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
#            print(detections.shape) # torch.Size([10, 7])
#            print(detections) # tensor([[5.4068e+02, 1.5408e+02, 5.6480e+02, 1.7346e+02, 9.9745e-01, 5.8307e-01, 5.8307e-01], ...)
            unique_labels = detections[:, -1].cpu().unique()
            #n_cls_preds = len(unique_labels)
            #bbox_colors = random.sample(colors, 3)
            for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):

#                print("\t+ Depth: %.5f, Conf: %.5f" % (float(cls_pred)*80, conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1
                #color = [float(cls_pred), float(cls_pred), float(cls_pred)]
                color = colors[i]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                
                plt.text(
                    x1,
                    y1,
                    s=format(float(cls_pred)*80, '.1f'),
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()

