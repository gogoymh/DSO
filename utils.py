from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
#    print(tp) 
#    print(pred_cls.shape)
#    print(target_cls) # length : 24,   [0.10043945163488388, 0.09829101711511612, ...
    # Sort by objectness
    i = np.argsort(-tp)
#    print(i)        # (365,) [ 65 269 171 116 299 104 332 303 173   0 305   6 166 270 334 105 108 224
                     #         215  57 214 338 223 117 170 339  58  63 333  71  67  62 262 106 167 216 ...
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
#    print('ap_per_class')
#    print(tp)
#    print(conf)
#    print(pred_cls)
    # Find unique classes
    unique_classes = np.unique(target_cls)
#    print(unique_classes) # not zero values in the variable 'tp'
    
    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
#    print('unique_classes = {}'.format(unique_classes))
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = abs(pred_cls - c) < 3/80 # boolean of predicted object 
        n_p = i.sum()  # Number of predicted objects
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
#        print('number of positives = {}\nnumber of ground truth = {}'.format(n_p, n_gt)) # number of positives = 5,  number of ground truth = 1
#        print(c) # 0.18056640028953552
#        print(i) # [ True False False False False False False False False False False True ...]        
        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()
#            print('tp[i] :\n{}'.format(tp[i])) # [1. 1. 0. 0. 0.]
#            print('fpc = {}\ntpc = {}'.format(fpc, tpc)) # fpc = [0. 0. 1. 2. 3.] tpc = [1. 2. 2. 2. 2.]


            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
#    print('compute_ap')
#    print(recall) # [1. 2. 2. 2. 2.]
#    print(precision) # [1.         1.         0.66666667 0.5        0.4       ]
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
#    print(mrec) # [0. 1. 2. 2. 2. 2. 1.]
#    print(mpre) # [0.         1.         1.         0.66666667 0.5        0.4         0.]
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
#    print('get_batch_statistics')
#    print(len(outputs)) # 1 (it's different as batch size)
#    print(outputs[0].shape) # size : (11, 7), tensor([[ 1.2910e+02,  1.9924e+02,  1.3787e+02,  2.0586e+02,  9.9114e-01, 8.3019e-01,  8.3019e-01], ...) 
#    print(targets.shape) # torch.Size([4, 6])
#    print(targets) # tensor([[0.0000e+00, 1.0576e-01, 6.3974e+01, 1.7242e+02, 8.2447e+01, 2.1655e+02], ...)  - [i-th image, label, x, y, w, h]
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue
        # output.shape: [:, 7]
        output = outputs[sample_i] # tensor([[2.6125e+02, 2.0466e+02, 2.6533e+02, 2.1460e+02, 8.5556e-01, 3.6729e-01, 3.6729e-01], ...)
        pred_boxes = output[:, :4] # means output[:, 0,1,2,3] is x,y,w,h
        pred_scores = output[:, 4] # means output[:, 4] is confidence
        pred_labels = output[:, -1] # means output[:, 6] is depth
#        print('pred_results : ')
#        print(pred_boxes.shape) # torch.Size([11, 4])
#        print(pred_boxes) # tensor([[129.0997, 199.2359, 137.8673, 205.8577], ...)
#        print(pred_labels.shape) # torch.Size([11])
#        print(pred_labels) # tensor([ 0.8302, ...)
        
        true_positives = np.zeros(pred_boxes.shape[0])
        false_positives = np.zeros(pred_boxes.shape[0])
        mean_depth_errors = np.zeros(pred_boxes.shape[0])
        relative_errors = np.zeros(pred_boxes.shape[0])        
#        print(true_positives.shape) # (74,)
        # target[:, 0] means it is about 'i'th image. 
        annotations = targets[targets[:, 0] == sample_i][:, 1:] # annotations is [x, y, w, h, label] target informations about a current image
#        print(annotations.shape) # torch.Size([7, 5])
#        print(annotations) # tensor([[1.0044e-01, 1.4711e+01, 1.8097e+02, 5.0289e+01, 2.2168e+02], ...)
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:] # target_boxes = target's [x,y,w,h]s

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break
#                print(annotations[:, 0].shape) # torch.Size([4])
#                print(annotations[:, 0]) # tensor([0.1004, 0.0983, 0.1806, 0.1763])
#                print(pred_label.shape) # torch.Size([]) 
#                print(pred_label) # tensor(0.2012)
#                print(target_labels.shape) # torch.Size([4])
#                print(target_labels) # tensor([0.1004, 0.0983, 0.1806, 0.1763])
                # Ignore if label is not one of the target labels. But do nothing in regression-problem
#                if pred_label not in target_labels:
#                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
#                print(target_labels) # tensor([0.1004, 0.0983, 0.1806, 0.1763])
                
#                print(target_labels[box_index]) # 0.1806
                if iou >= iou_threshold and box_index not in detected_boxes :#and abs(pred_label - target_labels[box_index]) < 1/80:
                #if iou >= iou_threshold and box_index not in detected_boxes:
#                    print('true positive')
#                    print(pred_i, box_index) # 0 tensor(1)
#                    print(pred_label)
#                    print(target_labels[box_index])
                    true_positives[pred_i] = 1# - abs(1 - (pred_label / target_labels[box_index]))
                    mean_depth_errors[pred_i] = abs(target_labels[box_index] - pred_label)
                    relative_errors[pred_i] = mean_depth_errors[pred_i] / target_labels[box_index]
                    detected_boxes += [box_index]
                else :
                    false_positives[pred_i] = 1
        batch_metrics.append([true_positives, pred_scores, pred_labels, mean_depth_errors, relative_errors, false_positives])
#        print(batch_metrics)
    return batch_metrics
#    batch_metrics : [true_positives, pred_scores, pred_labels]
#    print(true_positives) # [0. 0. 0. ... 0. 0. 0.]       //  shape : (158034,)
#    print(pred_scores.shape) # [0.9816881  0.502678   0.4432193  ... 0.0067421  0.00115292 0.00149798]  //  shape : (158034,)
#    print(pred_labels.shape) # [0.7118681  0.8186231  0.91287434 ... 0.07695463 0.38292742 0.19151904]  // shape : (158034,)

def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, conf_thres=0.7, nms_thres=0.5):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # prediction example : tensor([[[ 1.9009e+01,  2.1032e+00,  3.2086e+00,  5.0399e+00,  4.1740e-08, 7.4817e+00], ...]]) x, y, w, h, conf, cls
#    print('conf_thres = {}'.format(conf_thres))
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
#    print('prediction.shape : {}'.format(prediction.shape)) # torch.Size([1, 10647, 6])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):  
        # Filter out confidence scores below threshold
        
        image_pred = image_pred[image_pred[:, 4] >= conf_thres] # 10647 -> 27
#        print('image_pred.shape : {}'.format(image_pred.shape)) # torch.Size([27, 6])
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] # * image_pred[:, 5:].max(1)[0] 
        # Sort by it
        image_pred = image_pred[(-score).argsort()]

        
#        print(image_pred) # tensor([[2.3858e+02, 1.9849e+02, 2.4362e+02, 2.0973e+02, 8.4269e-01, 3.3781e-01], ...)
        
        
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True) # class_confs = class_preds (in regression)
        class_preds = image_pred[:, 5:] #######
        
        
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        
#        print('detections.shape : {}'.format(detections.shape)) # torch.Size([27, 7])
#        print(detections) # tensor([[1.8497e+02, 1.9971e+02, 1.9322e+02, 2.0634e+02, 9.9745e-01, 5.8307e-01, 5.8307e-01], ...)
        
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = abs(detections[0, -1] - detections[:, -1]) < 3/80
#            print(label_match)
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
    return output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
#    print('pred_boxes : {}'.format(pred_boxes.shape)) #    pred_boxes : torch.Size([8, 3, 15, 15, 4])
#    print('pred_cls.shape : {}'.format(pred_cls.shape))     #    pred_cls : torch.Size([8, 3, 15, 15, 1])
#    print('target.shape : {}'.format(target.shape))         #    target : torch.Size([60, 6])
    # batch index, depth, cx, cy, w, h
#    for i in range(target.shape[0]) :
#        print(target[i, :]) #tensor([6.0000, 0.1240, 0.5498, 0.2439, 0.0507, 0.0221], ... device='cuda:0')
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0) # 8
    nA = pred_boxes.size(1) # 3
    nC = pred_cls.size(-1) # 1 (torch.size(-1) returns a last dimension. (ex) A matrix 'a' has 2 rows ans 5 columns, then this function refers '5')
    nG = pred_boxes.size(2) # 15 (a size of anchor box)
    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
    #tcls = FloatTensor(nB, nA, nG, nG).fill_(0)
    #print("tcls.shape : {}".format(tcls.shape))

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]   # shape : torch.Size([67, 2])
    gwh = target_boxes[:, 2:]   # tensor([[1.2947, 1.1111], [0.5990, 0.4348], ...)
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
#    print('best_ious.shape : {}'.format(best_ious.shape))
#    print('best_n.shape : {}'.format(best_n.shape))
#    print('best_n : {}'.format(best_n)) 
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    
#    print('b.shape : {}'.format(b.shape))
#    print('b : {}'.format(b)) 
    #tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3,
    #    3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6,
    #    6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7], device='cuda:0')    
    #b = target[:, :1].long().t()
    
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
#    print('gi.shape : {}'.format(gi.shape))
#    print('gi : {}'.format(gi))
#    print('gj.shape : {}'.format(gj.shape))
#    print('gj : {}'.format(gj))
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    #tcls[b, best_n, gj, gi, target_labels] = 1
    #target_labels = target[:, 1:2].t()
#    print('target_labels.shape : {}'.format(target_labels.shape))
#    print('target_labels : {}'.format(target_labels))
#    
#    print('target[:, 1:2].t().shape : {}'.format(target[:, 1:2].t().shape))
#    print('target[:, 1:2].t() : {}'.format(target[:, 1:2].t()))
    #print('number of 1 values in target_labels : {}'.format(len(count)))
    tcls[b, best_n, gj, gi] = target[:, 1:2]
#    print('pred_cls.shape : {}'.format(pred_cls.shape))
#    print('pred_cls[b,best_n,gj,gi, target_labels] : {}'.format(pred_cls[b,best_n,gj,gi, target_labels]))
#    print('tcls.shape : {}'.format(tcls.shape))
#    print('tcls[b,best_n,gj,gi, target_labels] : {}'.format(tcls[b,best_n,gj,gi, target_labels]))
#    
#    print('pred_cls[b, best_n, gj, gi].argmax(-1) : {}'.format(pred_cls[b, best_n, gj, gi].argmax(-1)))
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
#    print('class_mask[b, best_n, gj, gi] : {}'.format(class_mask[b, best_n, gj, gi]))
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
#    print('obj_mask.shape : {}'.format(obj_mask.shape))
#    print('obj_mask[b, best_n, gj, gi] : {}'.format(obj_mask[b, best_n, gj, gi]))
#    
#    print('pred_cls[obj_masks] : {}'.format(pred_cls[obj_mask]))
#    print('tcls[obj_masks] : {}'.format(tcls[obj_mask]))
#    print('obj_mask : {}'.format(obj_mask))
#    print('pred_cls.shape : {}'.format(pred_cls.shape))
#    print('pred_cls : {}'.format(pred_cls))
#    print('tcls.shape : {}'.format(tcls.shape))
#    print('tcls : {}'.format(tcls))
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
