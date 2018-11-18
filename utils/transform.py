##===================== Statements & Copyright ===================##
"""
LOG:     April 12th
AUTHOR:  Xiaolong Li, VT
CONTENT: Used for Video-text project, reuse the LSTM model
"""
# Demo on using LSTM, tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import tensorflow as tf
import numpy as np
import xmltodict
from matplotlib import pyplot as plt
import cv2
import collections
import io
import platform
from datetime import datetime
from random import randint
import sys
import _init_paths

############ Macros ############
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION = "unidirection"

###############################
logging = tf.logging
now = datetime.now()


def convert_vect(line, w, h):
    text_lines = []
    object_num = 40
    for n in range(0, object_num):
        tl = collections.OrderedDict()
        tl['x0'] = line[n * 9]*w
        tl['y0'] = line[n * 9 + 1]*h
        tl['x1'] = line[n * 9 + 2]*w
        tl['y1'] = line[n * 9 + 3]*h
        tl['x2'] = line[n * 9 + 4]*w
        tl['y2'] = line[n * 9 + 5]*h
        tl['x3'] = line[n * 9 + 6]*w
        tl['y3'] = line[n * 9 + 7]*h
        text_lines.append(tl)
    dict_coded = {'text_lines':text_lines}
    return dict_coded


def read_xml_solo(filepath, frame):
    with open(filepath) as fd:
        doc = xmltodict.parse(fd.read())
    #print(doc['Frames']['frame'])
    #print(doc['Frames']['frame'][0])
    #print(doc['Frames']['frame'][0]['object'][0]['Point'][0]['@x'])
    #doc['Frames']['frame'][index]['@ID']
    #frame_num = len(doc['Frames']['frame'])
    i = frame
    print('%d th frame', i)
    text_lines = []
    if 'object' in doc['Frames']['frame'][i-1]:
        if '@Transcription' in doc['Frames']['frame'][i-1]['object']:
            object_num = 1
            tl = collections.OrderedDict()
            tl['x0'] = doc['Frames']['frame'][i - 1]['object']['Point'][0]['@x']
            tl['y0'] = doc['Frames']['frame'][i - 1]['object']['Point'][0]['@y']
            tl['x1'] = doc['Frames']['frame'][i - 1]['object']['Point'][1]['@x']
            tl['y1'] = doc['Frames']['frame'][i - 1]['object']['Point'][1]['@y']
            tl['x2'] = doc['Frames']['frame'][i - 1]['object']['Point'][2]['@x']
            tl['y2'] = doc['Frames']['frame'][i - 1]['object']['Point'][2]['@y']
            tl['x3'] = doc['Frames']['frame'][i - 1]['object']['Point'][3]['@x']
            tl['y3'] = doc['Frames']['frame'][i - 1]['object']['Point'][3]['@y']
            tl['ID'] = doc['Frames']['frame'][i - 1]['object']['@ID']
            tl['score'] = 1
            text_lines.append(tl)
        else:
            object_num = len(doc['Frames']['frame'][i-1]['object'])
            for j in range(0, object_num):
                tl = collections.OrderedDict()
                tl['x0'] = doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@x']
                tl['y0'] = doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@y']
                tl['x1'] = doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@x']
                tl['y1'] = doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@y']
                tl['x2'] = doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@x']
                tl['y2'] = doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@y']
                tl['x3'] = doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@x']
                tl['y3'] = doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@y']
                tl['ID'] = doc['Frames']['frame'][i-1]['object'][j]['@ID']
                tl['score'] = 1
                text_lines.append(tl)
    target= {'text_lines': text_lines, }
    return target


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
