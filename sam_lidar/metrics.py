import numpy as np
from .utils import hungarian_matching
    

def per_tree_metrics(truths, preds, detection_threshold=0.5, segmentation_threshold=0.5):
    '''
    Calculate object-based and pixel-based precision, recall, and F1 across all trees in one or more images.
    This means if one image has 1 tree and another image has 10 trees, metrics are calculated across all 11
    trees rather than separately on the 1 tree and the 10 trees and then averaged together.

    params:
        truths (list): List of sv.Detections objects. Each Detections object includes the ground truth bounding boxes 
                       and masks for a single image.
        preds (list): List of sv.Detections objects. Each Detections object includes the prompt bounding boxes and
                      predicted masks for a single image.
        detection_threshold (float): Minimum IoU between predicted and ground truth masks to count as a True Positive
                                     for detection metrics
        segmentation_threshold (float): Minimum IoU between predicted and ground truth boxes to include a prediction in
                                        the segmentation metrics
    
    return:
        (tuple) object-based precision, recall, and f1, pixel-based precision, recall, f1, and iou
    '''
    # List of ground truths and list of predictions should be same length (i.e. same number of images with
    # ground truth labels and predicted labels)
    assert len(truths) == len(preds)

    object_tp = []
    object_fp = []
    object_fn = []
    pixel_precision = []
    pixel_recall = []
    pixel_f1 = []
    pixel_iou = []

    # Iterate thru each image
    for i in range(len(truths)):
        true_boxes = truths[i].xyxy
        true_masks = truths[i].mask
        pred_boxes = preds[i].xyxy
        pred_masks = preds[i].mask

        # Match ground truth and predicted masks for given image above detection_threshold
        true_idx, pred_idx = hungarian_matching(true_masks, pred_masks, threshold=detection_threshold)

        # Record detection (tree-wise) TP, FP, and FN
        object_tp.append(len(true_idx))
        object_fp.append(len(pred_boxes) - len(pred_idx))
        object_fn.append(len(true_boxes) - len(true_idx))

        # Match ground truth and predicted boxes for given image above segmentation_threshold
        # to select corresponding masks to run segmentation metrics on
        pixel_true_idx, pixel_pred_idx = hungarian_matching(true_boxes, pred_boxes, threshold=segmentation_threshold)
        true_masks = true_masks[pixel_true_idx]
        pred_masks = pred_masks[pixel_pred_idx]

        # Calculate segmentation (pixel-wise) TP, FP, and FN
        pixel_tp = np.logical_and(true_masks, pred_masks).sum(axis=(1,2))
        pixel_fp = np.logical_and(np.logical_not(true_masks), pred_masks).sum(axis=(1,2))
        pixel_fn = np.logical_and(true_masks, np.logical_not(pred_masks)).sum(axis=(1,2))

        # Use pixel-wise TP, FP, and FN to calculate tree-wise segmentation metrics
        precision = np.divide(pixel_tp, (pixel_tp+pixel_fp), out=np.zeros(pixel_tp.shape), where=(pixel_tp+pixel_fp)!=0)
        recall = np.divide(pixel_tp, (pixel_tp+pixel_fn), out=np.zeros(pixel_tp.shape), where=(pixel_tp+pixel_fn)!=0)
        f1 = np.divide((2*precision*recall), (precision+recall), out=np.zeros(recall.shape), where=(precision+recall)!=0)
        iou = np.divide(pixel_tp, (pixel_tp+pixel_fp+pixel_fn), out=np.zeros(pixel_tp.shape), where=(pixel_tp+pixel_fp+pixel_fn)!=0)
        pixel_precision += list(precision)
        pixel_recall += list(recall)
        pixel_f1 += list(f1)
        pixel_iou += list(iou)

    # Calculate detection precision, recall, and f1 for all trees across all images
    object_tp = sum(object_tp)
    object_fp = sum(object_fp)
    object_fn = sum(object_fn)
    object_precision = object_tp / (object_tp + object_fp) if (object_tp + object_fp) > 0 else 0
    object_recall = object_tp / (object_tp + object_fn) if (object_tp + object_fn) > 0 else 0
    object_f1 = (2 * object_precision * object_recall) / (object_precision + object_recall) if (object_precision + object_recall) > 0 else 0

    # Average segmentation precision, recall, f1, and IoU for all trees across all images
    pixel_precision = sum(pixel_precision) / len(pixel_precision) if len(pixel_precision) > 0 else 0
    pixel_recall = sum(pixel_recall) / len(pixel_recall) if len(pixel_recall) > 0 else 0
    pixel_f1 = sum(pixel_f1) / len(pixel_f1) if len(pixel_f1) > 0 else 0
    pixel_iou = sum(pixel_iou) / len(pixel_iou) if len(pixel_iou) > 0 else 0

    return np.array([object_precision, object_recall, object_f1,
                     pixel_precision, pixel_recall, pixel_f1, pixel_iou])
    

def per_tree_std(truths, preds, segmentation_threshold=0.5):
    '''
    Calculate standard deviation of pixel-based metrics across all trees in one or more images.
    This means if one image has 1 tree and another image has 10 trees, standard deviation is calculated 
    across all 11 trees rather than separately on the 1 tree and the 10 trees and then averaged together.

    params:
        truths (list): List of sv.Detections objects. Each Detections object includes the ground truth bounding boxes 
                       and masks for a single image.
        preds (list): List of sv.Detections objects. Each Detections object includes the prompt bounding boxes and
                      predicted masks for a single image.
        segmentation_threshold (float): Minimum IoU between predicted and ground truth boxes to include a prediction in
                                        the segmentation metrics
    
    return:
        (tuple) Standard deviation of pixel-based precision, recall, f1, and iou across all trees.
    '''
    # List of ground truths and list of predictions should be same length (i.e. same number of images with
    # ground truth labels and predicted labels)
    assert len(truths) == len(preds)
    
    pixel_precision = []
    pixel_recall = []
    pixel_f1 = []
    pixel_iou = []

    for i in range(len(truths)):
        true_boxes = truths[i].xyxy
        true_masks = truths[i].mask
        pred_boxes = preds[i].xyxy
        pred_masks = preds[i].mask

        # Match ground truth and predicted boxes for given image above segmentation_threshold
        # to select corresponding masks to run segmentation metrics on
        pixel_true_idx, pixel_pred_idx = hungarian_matching(true_boxes, pred_boxes, threshold=segmentation_threshold)
        true_masks = true_masks[pixel_true_idx]
        pred_masks = pred_masks[pixel_pred_idx]

        # Calculate segmentation (pixel-wise) TP, FP, and FN
        pixel_tp = np.logical_and(true_masks, pred_masks).sum(axis=(1,2))
        pixel_fp = np.logical_and(np.logical_not(true_masks), pred_masks).sum(axis=(1,2))
        pixel_fn = np.logical_and(true_masks, np.logical_not(pred_masks)).sum(axis=(1,2))

        # Use pixel-wise TP, FP, and FN to calculate tree-wise segmentation metrics
        precision = np.divide(pixel_tp, (pixel_tp+pixel_fp), out=np.zeros(pixel_tp.shape), where=(pixel_tp+pixel_fp)!=0)
        recall = np.divide(pixel_tp, (pixel_tp+pixel_fn), out=np.zeros(pixel_tp.shape), where=(pixel_tp+pixel_fn)!=0)
        f1 = np.divide((2*precision*recall), (precision+recall), out=np.zeros(recall.shape), where=(precision+recall)!=0)
        iou = np.divide(pixel_tp, (pixel_tp+pixel_fp+pixel_fn), out=np.zeros(pixel_tp.shape), where=(pixel_tp+pixel_fp+pixel_fn)!=0)
        pixel_precision += list(precision)
        pixel_recall += list(recall)
        pixel_f1 += list(f1)
        pixel_iou += list(iou)

    return np.std(pixel_precision), np.std(pixel_recall), np.std(pixel_f1), np.std(pixel_iou)


def box_mask_metrics(truths, preds, detection_threshold=0.5, segmentation_threshold=0.5):
    '''
    Calculate object-based metrics for both bounding boxes and masks. Used for an experiment in the
    paper comparing both.

    params:
        truths (list): List of sv.Detections objects. Each Detections object includes the ground truth
                    bounding boxes and masks for a single image.
        preds (list): List of sv.Detections objects. Each Detections object includes the predicted
                    bounding boxes and masks for a single image.
        detection_threshold (float): Minimum IoU between predicted and ground truth masks to count
                                    as a True Positive for detection metrics
        segmentation_threshold (float): Minimum IoU between predicted and ground truth boxes to include
                                        a prediction in the segmentation metrics
    returns:
        bounding box precision, recall, and f1, mask precision, recall, and f1 (tuple)
    '''
    # List of ground truths and list of predictions should be same length (i.e. same number of images with
    # ground truth labels and predicted labels)
    assert len(truths) == len(preds)

    box_tp, box_fp, box_fn = [],[],[]
    mask_tp, mask_fp, mask_fn = [],[],[]

    # Iterate thru each image
    for i in range(len(truths)):
        true_boxes = truths[i].xyxy
        true_masks = truths[i].mask
        pred_boxes = preds[i].xyxy
        pred_masks = preds[i].mask

        # Match ground truth and predicted boxes for given image above detection_threshold
        true_idx, box_idx = hungarian_matching(true_boxes, pred_boxes, threshold=detection_threshold)

        # Record box (tree-wise) TP, FP, and FN
        box_tp.append(len(true_idx))
        box_fp.append(len(pred_boxes) - len(box_idx))
        box_fn.append(len(true_boxes) - len(true_idx))

        # Match ground truth and predicted masks for given image above detection_threshold
        true_idx, mask_idx = hungarian_matching(true_masks, pred_masks, threshold=detection_threshold)

        # Record mask (tree-wise) TP, FP, and FN
        mask_tp.append(len(true_idx))
        mask_fp.append(len(pred_masks) - len(mask_idx))
        mask_fn.append(len(true_masks) - len(true_idx))

    # Calculate box and mask precision, recall, and f1 for all trees across all images
    box_tp = sum(box_tp)
    box_fp = sum(box_fp)
    box_fn = sum(box_fn)

    mask_tp = sum(mask_tp)
    mask_fp = sum(mask_fp)
    mask_fn = sum(mask_fn)   

    box_precision = box_tp / (box_tp + box_fp) if (box_tp + box_fp) > 0 else 0
    box_recall = box_tp / (box_tp + box_fn) if (box_tp + box_fn) > 0 else 0
    box_f1 = (2 * box_precision * box_recall) / (box_precision + box_recall) if (box_precision + box_recall) > 0 else 0

    mask_precision = mask_tp / (mask_tp + mask_fp) if (mask_tp + mask_fp) > 0 else 0
    mask_recall = mask_tp / (mask_tp + mask_fn) if (mask_tp + mask_fn) > 0 else 0
    mask_f1 = (2 * mask_precision * mask_recall) / (mask_precision + mask_recall) if (mask_precision + mask_recall) > 0 else 0
 

    return np.array([box_precision, box_recall, box_f1,
                     mask_precision, mask_recall, mask_f1])