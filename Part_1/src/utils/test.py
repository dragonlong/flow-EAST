import cv2
import os
import time
import numpy as np
import logging
import collections
import xmltodict

from eval import resize_image, sort_poly, detect

def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, thickness=2, color=(255, 255, 0))
    return illu


def draw_illu_gt(illu, rst, p, r, f):
    if 'text_lines' in rst:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        fontColor = (255, 255, 255)
        lineType = 1
        infos = 'Precision ' + str(p)+ ', recall ' + str(r) + ', F_measure ' + str(f)
        cv2.putText(illu, infos,
                    (2, 20),
                    font,
                    0.5,
                    (255, 0, 0),
                    lineType)
        for t in rst['text_lines']:
            d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                          t['y2'], t['x3'], t['y3']], dtype='int32')
            d = d.reshape(-1, 2)
            cv2.polylines(illu, [d], isClosed=True, thickness=2, color=(0, 0, 0))
            bottomLeftCornerOfText = (int(t['x0']), int(t['y0']))
            cv2.putText(illu, t['ID'],
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
    return illu

if __name__ == '__main__':
    basename = '/home/xiaolong/Downloads/pre-processed'
    img = np.load(basename + '/frame000_img.npy').astype(np.uint8)
    sco = np.load(basename + '/frame000_sco.npy')
    geo = np.load(basename + '/frame000_geo.npy')
    print('with shape {}'.format(img.shape))
    timer = collections.OrderedDict([
    ('net', 0),
    ('restore', 0),
    ('nms', 0)])
    boxes, timer = detect(score_map=sco[::4, ::4], geo_map=geo[::4, ::4, :], timer=timer)
    if boxes is not None:
        scores = boxes[:,8].reshape(-1)
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= 1
        boxes[:, :, 1] /= 1

    text_lines = []
    if boxes is not None:
        for box, score in zip(boxes, scores):
            box = sort_poly(box.astype(np.int32))
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                continue
            tl = collections.OrderedDict(zip(
                ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                map(float, box.flatten())))
            tl['score'] = float(score)
            text_lines.append(tl)
        ret = {
            'text_lines': text_lines,
            # 'rtparams': rtparams,
            # 'timing': timer,
            # 'geometry': geometry,
            # 'score':float(score),
        }
            # 1. print boxes number
        print('%d Boxs found'%(len(text_lines)))
        new_img = draw_illu(img.copy(), ret)
        cv2.imshow('Annotated Frame with EAST', new_img)
