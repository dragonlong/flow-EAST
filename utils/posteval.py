from utils.nms_highlevel import intersection
import tensorflow as tf
from utils import icdar
import numpy as np


# used for iou searching and selecting TP, FP, FN #
def eval_single(bb_gt, bb_pred):
    """
    input params:
        bb_gt, numpy array, (batch_size, time_steps, 40*9)
        bb_pred, ((batch_size, time_steps, 40*9))
    """
    length_vect = bb_gt.shape[2]
    unit = 9
    num  = int(length_vect/unit)
    is_best = 0
    TP   = 0
    FP   = 0
    FN   = 0
    for i in range(num):
        for j in range(num):
            # pick out the best match
            iou = intersection(bb_gt[:,:,i*unit:i*unit+8], bb_pred[:,:,i*unit:i*unit+8])
            if iou>is_best:
                is_best = iou
        if iou > 0.5:
            TP = TP+1
        elif iou > 0:
            FP = FP+1
        else:
            FN = FN+1
    precision = TP/(TP+FP)
    recall    = TP/(TP+FN)
    F_measure = 2*precision*recall/(precision+recall)
    return precision, recall, F_measure


def eval_group(bb_gt, bb_pred):
    """
    
    """
    [m, n, _] = bb_gt.shape
    prec_c    = np.zeros((m, n), dtype=np.float32)
    recall_c  = np.zeros((m, n), dtype=np.float32)
    fmea_c    = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            p, r, f = eval_single(bb_gt[i, j, :], bb_pred[i, j, :])
            prec_c[i, j]   = p
            recall_c[i, j] = r
            fmea_c[i, j]   = f
    return prec_c, recall_c, fmea_c


# used for iou searching and selecting TP, FP, FN #
def eval_single_frame(target, box):
    """
    input params:
        target, python ordered dict
        box, sorted boxes dict from predictions
    """
    TP   = 0
    FP   = 0
    FN   = 0
    precision = 0
    recall = 0
    F_measure = 0
    if not len(target['text_lines'])==0:
        if not len(box['text_lines']) == 0:
            for t in target['text_lines']:
                d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                              t['y2'], t['x3'], t['y3']], dtype='int32')
                is_best = 0
                for m in box['text_lines']:
                    n = np.array([m['x0'], m['y0'], m['x1'], m['y1'], m['x2'],
                                  m['y2'], m['x3'], m['y3']], dtype='int32')

                    # pick out the best match
                    iou = intersection(n, d)
                    if iou>is_best:
                        is_best = iou
                if is_best > 0.5:
                    TP = TP+1
                elif is_best == 0:
                    FN = FN +1
                else:
                    FP = FP+1
            if TP > 0:
                precision = TP/(TP+FP)
                recall    = TP/(TP+FN)
                F_measure = 2*precision*recall/(precision+recall)
    return precision, recall, F_measure

