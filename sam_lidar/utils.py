import numpy as np
from supervision.detection.utils import box_iou_batch
from scipy.optimize import linear_sum_assignment


def box_area(box):
    t_box = box.T
    return (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])


def intersection_over_minimum_area(boxes):
    '''
    Similar to IoU, but instead of dividing the area of the intersection by
    the area of the union, divide it by the area of the smaller box.

    params:
        boxes (ndarray): N x 4 array of boxes in xyxy format. The intersection
                         over minimum area of each box in the array will be
                         calculated with every other box in the array.

    returns:
        inter_over_min_area (ndarray): N x N array giving the intersection over
                                       minimum area of every box with every
                                       other box.
    '''
    areas = box_area(boxes)

    top_left = np.maximum(boxes[:, None, :2], boxes[:, :2])
    bottom_right = np.minimum(boxes[:, None, 2:], boxes[:, 2:])

    inter_area = np.prod(
    	np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    inter_over_area_1 = inter_area / areas[:, None]
    inter_over_area_2 = inter_area / areas[None, :]
    inter_over_min_area = np.maximum(inter_over_area_1, inter_over_area_2)

    return inter_over_min_area


def custom_nms(detections, threshold = 0.75, ignore_categories = False):
    '''
    Like regular Non-Max Suppression, but instead of intersection over union (IoU), measures
    intersection over the area of the smaller box. This means if e.g. a smaller box is completely
    contained in a box twice its size, while the IoU would be 0.5, the intersection over minimum
    area would be 1.0, because the area of the smaller box is equal to the area of the intersection.

    params:
        detections:   Supervision Detections object with xyxy, confidence, and class_id fields.

         threshold:  The percent of a given box's total area that is contained within another box
                     to activate custom non-max suppression.

        ignore_categories:  If False, external_box_suppression will only be applied to boxes with the
                            same label, otherwise will be applied between all boxes regardless of label.

    returns:
        keep: Index of the detections to keep after non-max suppression.
    '''
    boxes = detections.xyxy
    confidence = detections.confidence
    categories = detections.class_id

    rows = boxes.shape[0]

    sort_index = np.flip(confidence.argsort())
    boxes = boxes[sort_index]
    confidence = confidence[sort_index]
    categories = categories[sort_index]

    overlaps = intersection_over_minimum_area(boxes)
    overlaps = overlaps - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (overlap, category) in enumerate(zip(overlaps, categories)):
        if not keep[index]:
            continue

        if ignore_categories:
            condition = (overlap > threshold)
        else:
            condition = (overlap > threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]


def mask_iou_batch(masks_true, masks_pred):
    '''
    Calculate IoU between all pairs of two batches of masks, return matrix of IoUs.
    '''
    intersection = np.logical_and(masks_true[:,None,...], masks_pred).sum(axis=(2,3))
    union = np.logical_or(masks_true[:,None,...], masks_pred).sum(axis=(2,3))
    iou_matrix = np.divide(intersection, union, out=np.zeros(union.shape), where=union!=0)
    return iou_matrix


def boxes_to_masks(boxes, shape):
    '''
    Convert bounding boxes to masks for purposes of computing IoU between boxes and masks.
    '''
    masks = []
    for i, box in enumerate(boxes):
        masks.append(np.full(shape, False))
        masks[i][round(box[1]):round(box[3])+1, round(box[0]):round(box[2])+1] = True
    if len(masks) > 0:
        return np.stack(masks)
    else:
        return np.empty(shape)[np.newaxis]

def hungarian_matching(truths, preds, threshold=0.0):
    '''
    Compute Hungarian Matching between ground truth masks or bounding boxes
    and predicted masks or bounding boxes.

    parameters:
        truths (array): Ground truth bounding boxes or masks
        preds (array): Predicted bounding boxes or masks
        threshold (float): Only return indexes with an IoU greater than the given threshold

    returns:
        true_idx, pred_idx (floats): corresponding indices for the truths and preds with greatest IoU
    '''
    # If both preds and truths are bounding boxes, calculate IoU between boxes and feed into hungarian algorithm
    if truths.ndim == 2 and preds.ndim == 2:
        iou_matrix = box_iou_batch(truths, preds)
        true_idx, pred_idx = linear_sum_assignment(iou_matrix, maximize=True)
        idx = np.array([i for i in range(len(true_idx)) if iou_matrix[true_idx[i], pred_idx[i]] >= threshold], dtype='int')
        return true_idx[idx], pred_idx[idx]

    # If only one is bounding boxes, convert to mask to compute IoU
    elif truths.ndim == 2:
        truths = boxes_to_masks(truths, preds.shape[1:])

    elif preds.ndim == 2:
        preds = boxes_to_masks(preds, truths.shape[1:])

    # Calculate IoU between masks, feed into hungarian algorithm
    iou_matrix = mask_iou_batch(truths, preds)
    true_idx, pred_idx = linear_sum_assignment(iou_matrix, maximize=True)
    idx = np.array([i for i in range(len(true_idx)) if iou_matrix[true_idx[i], pred_idx[i]] >= threshold], dtype='int')
    return true_idx[idx], pred_idx[idx]


def compute_iou(truths, preds, reduction='none'):
    '''
    Calculate IoU between matched masks, return all or average IoU.

    params:
        truths (ndarray): Numpy array of shape (T,H,W) for ground truth masks, where T is the number of identified trees, 
                          H and W are the height and width of the image.
        preds (ndarray): Numpy array of shape (T,H,W) for predicted masks, where T is the number of identified trees,
                         H and W are the height and width of the image.
        reduction (str): If 'none', return metrics for each tree, if "mean" return average metrics.
    
    return:
        IoU (ndarray)
    '''
    intersection = np.logical_and(truths, preds).sum(axis=(1,2))
    union = np.logical_or(truths, preds).sum(axis=(1,2))
    iou = np.divide(intersection, union, out=np.zeros(union.shape), where=union!=0)


    # If reduction is 'mean', return the mean IoU, otherwise return all IoUs
    if reduction == 'none':
        return iou
    elif reduction == 'mean':
        return np.mean(iou)
    else:
        raise ValueError(f"reduction should be either 'mean' or 'none'.")