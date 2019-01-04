  # Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import xmltodict
import numpy as np
import json
import cv2
import time
import platform
from skimage import measure
from shapely.geometry import Polygon
import tensorflow as tf
from scipy import ndimage, misc
# from pdb import set_trace as pb
# sys.path.append('..')
from utils.icdar import shrink_poly, fit_line, point_dist_to_line, line_cross_point, rectangle_from_parallelogram, sort_rectangle, check_and_validate_polys
Py3 = sys.version_info[0] == 3
tf.app.flags.DEFINE_string('raw_path', '/media/dragonx/DataStorage/temporary/Video_text/ICDAR/train/', 'for data import')
FLAGS = tf.app.flags.FLAGS


def read_XML(filepath, f_start, f_end, cap=40, f_w=720, f_h=480):
    with open(filepath+'XML/Video_16_3_2_GT.xml') as fd:
        doc = xmltodict.parse(fd.read())
    #print(doc['Frames']['frame'])
    #print(doc['Frames']['frame'][0])
    #print(doc['Frames']['frame'][0]['object'][0]['Point'][0]['@x'])
    #doc['Frames']['frame'][index]['@ID']
    #frame_num = len(doc['Frames']['frame'])
    frame_num = f_end - f_start+1
    target = np.zeros((frame_num, 9*cap))
    for i in range(f_start, f_end+1):
        print('%d th frame', i)
        object_num = len(doc['Frames']['frame'][i-1]['object'])
        for j in range(0, object_num):
            print(j)
            target[i - f_start, j*9]   = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@x'])/f_w
            target[i - f_start, j*9+1] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@y'])/f_h
            target[i - f_start, j*9+2] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@x'])/f_w
            target[i - f_start, j*9+3] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@y'])/f_h
            target[i - f_start, j*9+4] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@x'])/f_w
            target[i - f_start, j*9+5] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@y'])/f_h
            target[i - f_start, j*9+6] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@x'])/f_w
            target[i - f_start, j*9+7] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@y'])/f_h
            target[i - f_start, j*9+8] = 1.0
    return target


def read_XML_solo(filepath, f_start, f_end, cap=40, f_w=720, f_h=480):
    with open(filepath+'_GT.xml') as fd:
        doc = xmltodict.parse(fd.read())
    frame_num = f_end- f_start+1
    target = np.zeros((frame_num, 9*cap))
    for i in range(f_start, f_end+1):
        #print('%d th frame', i)
        if 'object' in doc['Frames']['frame'][i - 1]:
            if '@Transcription' in doc['Frames']['frame'][i - 1]['object']:
                j = 0
                target[i - f_start, j * 9] = int(doc['Frames']['frame'][i - 1]['object']['Point'][0]['@x'])/f_w
                target[i - f_start, j * 9+1] = int(doc['Frames']['frame'][i - 1]['object']['Point'][0]['@y'])/f_h
                target[i - f_start, j * 9+2] = int(doc['Frames']['frame'][i - 1]['object']['Point'][1]['@x'])/f_w
                target[i - f_start, j * 9+3] = int(doc['Frames']['frame'][i - 1]['object']['Point'][1]['@y'])/f_h
                target[i - f_start, j * 9+4] = int(doc['Frames']['frame'][i - 1]['object']['Point'][2]['@x'])/f_w
                target[i - f_start, j * 9+5] = int(doc['Frames']['frame'][i - 1]['object']['Point'][2]['@y'])/f_h
                target[i - f_start, j * 9+6] = int(doc['Frames']['frame'][i - 1]['object']['Point'][3]['@x'])/f_w
                target[i - f_start, j * 9+7] = int(doc['Frames']['frame'][i - 1]['object']['Point'][3]['@y'])/f_h
                target[i - f_start, j * 9+8] = 1.0
            else:
                object_num = len(doc['Frames']['frame'][i-1]['object'])
                for j in range(0, object_num):
                    #print(j)
                    target[i - f_start, j*9]   = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@x'])/f_w
                    target[i - f_start, j*9+1] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@y'])/f_h
                    target[i - f_start, j*9+2] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@x'])/f_w
                    target[i - f_start, j*9+3] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@y'])/f_h
                    target[i - f_start, j*9+4] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@x'])/f_w
                    target[i - f_start, j*9+5] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@y'])/f_h
                    target[i - f_start, j*9+6] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@x'])/f_w
                    target[i - f_start, j*9+7] = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@y'])/f_h
                    target[i - f_start, j*9+8] = 1.0
    return target


def load_annotations_solo(filepath, f_start, f_end, f_w=160, f_h=160):
    with open(filepath+'_GT.xml') as fd:
        doc = xmltodict.parse(fd.read())
    frame_num = f_end- f_start+1
    polys_array_list = []
    tags_array_list = []
    id_list_list = []
    for i in range(f_start, f_end+1):
        text_polys = []
        tags = []
        id_container = []
        if 'object' in doc['Frames']['frame'][i - 1]:
            if '@Transcription' in doc['Frames']['frame'][i - 1]['object']:
                x1 = int(doc['Frames']['frame'][i - 1]['object']['Point'][0]['@x'])/f_w*512
                y1 = int(doc['Frames']['frame'][i - 1]['object']['Point'][0]['@y'])/f_h*512
                x2 = int(doc['Frames']['frame'][i - 1]['object']['Point'][1]['@x'])/f_w*512
                y2 = int(doc['Frames']['frame'][i - 1]['object']['Point'][1]['@y'])/f_h*512
                x3 = int(doc['Frames']['frame'][i - 1]['object']['Point'][2]['@x'])/f_w*512
                y3 = int(doc['Frames']['frame'][i - 1]['object']['Point'][2]['@y'])/f_h*512
                x4 = int(doc['Frames']['frame'][i - 1]['object']['Point'][3]['@x'])/f_w*512
                y4 = int(doc['Frames']['frame'][i - 1]['object']['Point'][3]['@y'])/f_h*512
                text_id = int(doc['Frames']['frame'][i-1]['object']['@ID'])
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                tags.append(True)
                id_container.append(text_id)
            else:
                object_num = len(doc['Frames']['frame'][i-1]['object'])
                for j in range(0, object_num):
                    x1 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@x'])/f_w*512
                    y1 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@y'])/f_h*512
                    x2 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@x'])/f_w*512
                    y2 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@y'])/f_h*512
                    x3 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@x'])/f_w*512
                    y3 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@y'])/f_h*512
                    x4 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@x'])/f_w*512
                    y4 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@y'])/f_h*512
                    text_id = int(doc['Frames']['frame'][i - 1]['object'][j]['@ID'])
                    text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    tags.append(True)
                    id_container.append(text_id)
        polys_array_list.append(np.array(text_polys, dtype=np.float32))
        tags_array_list.append(np.array(tags, dtype=bool))
        id_list_list.append(np.array(id_container, dtype=np.int16))

    return polys_array_list, tags_array_list, id_list_list, frame_num


def read_json_npy(filepath, index):
    with open(filepath+'json/frame'+format(index, '03d')+'.json') as f:
        d = json.load(f)
    data = np.load(filepath+'npy/frame'+format(index, '03d')+'.npy')
    return d, data


def read_heatmap(video_solo_path, w_d = 512, h_d = 512):
    ulti_path = video_solo_path + '/score'
    #num = len([name for name in os.listdir(ulti_path) if os.path.isfile(name)])
    num = len(os.listdir(ulti_path))
    print(num)
    heatmap_set = np.zeros((num, w_d, h_d))
    heatmap_vect = np.zeros((num, w_d*h_d), dtype=np.float32)
    for index in range(num):
        data =np.squeeze(np.load(ulti_path+'/frame' + format(index+1, '03d') + '.npy'))
        data_fit = cv2.resize(data, (w_d, h_d))
        heatmap_set[index, :, :] = data_fit
        heatmap_vect[index, :] = data_fit.flatten()

    return heatmap_set, heatmap_vect, num


def fusion_data(d, data_shrink, cap=15, f_w=720, f_h=480):
    d1 = np.ndarray.flatten(data_shrink)
    # with default 40 at most
    dim = cap*9
    output = np.zeros((len(d1)+dim),dtype=np.float32)
    for i in range(0, len(d['text_lines'])):
        output[i*9] = d['text_lines'][i]['x0']/f_w
        output[i*9+1] = d['text_lines'][i]['y0']/f_h
        output[i*9+2] = d['text_lines'][i]['x1']/f_w
        output[i*9+3] = d['text_lines'][i]['y1']/f_h
        output[i*9+4] = d['text_lines'][i]['x2']/f_w
        output[i*9+5] = d['text_lines'][i]['y2']/f_h
        output[i*9+6] = d['text_lines'][i]['x3']/f_w
        output[i*9+7] = d['text_lines'][i]['y3']/f_h
        output[i*9+8] = d['text_lines'][i]['score']
    output[dim:] = d1

    return output


def vect_producer(datapath, frame_start, frame_end, batch_size, num_steps, name=None):
    d, data = read_json_npy(datapath, frame_start)
    data_shrink = measure.block_reduce(np.squeeze(data), (10, 10, 1), np.mean)
    vect_encoded = fusion_data(d, data_shrink)
    l1 = len(vect_encoded)
    frame_num = frame_end - frame_start+1
    vect_set = np.zeros((frame_num, l1))
    vect_set[0, :] = vect_encoded
    for index in range(frame_start+1, frame_end+1):
        d, data = read_json_npy(datapath, index)
        data_shrink  = measure.block_reduce(np.squeeze(data), (10, 10, 1), np.mean)
        vect_encoded = fusion_data(d, data_shrink)
        vect_set[index-frame_start, :] = vect_encoded
    target = read_XML(datapath, frame_start, frame_end)
    _, l2 = target.shape
    print(vect_set.shape)
    print(target.shape)
    # convert to tensor and using shuffle queue
    with tf.name_scope(name, 'VectProducer', [batch_size, num_steps]):
        input_t = tf.convert_to_tensor(vect_set, name="input", dtype=tf.float32)
        target_t = tf.convert_to_tensor(target, name="target", dtype=tf.float32)
        batch_len  = frame_num // batch_size
        epoch_size = (batch_len - 1) // num_steps
        # parsing data from [frame_num, vect_length] to [batch_size, num_steps, vocal_size ]
        data_in = tf.reshape(input_t, [batch_size, batch_len, l1])
        data_gt = tf.reshape(target_t, [batch_size, batch_len, l2])
        # produce an iterable object for index
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data_in, [0, i * num_steps, 0], [batch_size, (i + 1) * num_steps, l1])
        x.set_shape([batch_size, num_steps, l1])
        y = tf.strided_slice(data_gt, [0, i * num_steps, 0], [batch_size, (i + 1) * num_steps, l2])
        y.set_shape([batch_size, num_steps, l2])
        return x, y
        # reshape


############## first divide the video, 8, 2, 2 as train, validate, test##############
# return tuple contains index for video, according to cnt_frame, we decide the starting frame index in random
def vect_producer_mul(datapath, video_start,  video_end):
    # first know hwo many videos we have
    """
    :param datapath:
    :param video_start:
    :param video_end:
    :return: list of
    """
    folder_list = []
    size_fixed = 160
    raw_path = '/media/dragonx/DataStorage/temporary/Video_text/ICDAR/train/'
    if platform.uname()[1] != "dragonx-H97N-WIFI":
        raw_path = '/home/lxiaol9/ARC/EASTRNN/data/ICDAR/train/'
        datapath = "/home/lxiaol9/ARC/EASTRNN/data/GAP_process"
#     for dir_name, dir_names, file_names in os.walk(datapath):
#         for sub_dir_name in dir_names:
#             folder_list.append(sub_dir_name)
    cnt_video = 12
    video_set = ['Video_37_2_3', 'Video_51_7_4', 'Video_33_2_3', 'Video_47_6_4', 'Video_28_1_1', 'Video_40_2_3', 'Video_4_4_2', 'Video_42_2_3', 'Video_21_5_1', 'Video_54_7_4', 'Video_45_6_4', 'Video_16_3_2']
    print(video_set)
    frame_info = []
    video_name = []
    heat_vect_collect = []
    target_collect = []
    for k in range(video_start-1, video_end):
        t1 = time.time()
        video_name.append(video_set[k])
        video_solo_path = datapath + '/' + video_set[k]
        # sort up all the paths
        xml_solo_path = raw_path+video_set[k]
        raw_video_path = raw_path + video_set[k]+'.mp4'
        cap = cv2.VideoCapture(raw_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # cnt_frame is frame number, heat_map_vect in (frame, w_d*h_d)
        _, heat_map_vect, cnt_frame = read_heatmap(video_solo_path, w_d=size_fixed, h_d=size_fixed)
        target = read_XML_solo(xml_solo_path, 1, cnt_frame, f_w=frame_width, f_h=frame_height)
        frame_info.append(cnt_frame)
        heat_vect_collect.append(heat_map_vect)
        target_collect.append(target)
        print('processed %d th video %s, taking %f seconds' %(k, video_set[k], time.time()-t1))
        # parsing data from [frame_num, vect_length] to [batch_size, num_steps, vocal_size ]

    return heat_vect_collect, target_collect, frame_info, video_name


############## first divide the video, 8, 2, 2 as train, validate, test##############
# return tuple contains index for video, according to cnt_frame, we decide the starting frame index in random
def array_producer_mul(datapath, video_start,  video_end):
    # first know hwo many videos we have
    """
    :param datapath:
    :param video_start:
    :param video_end:
    :return: list of
    """
    folder_list = []
    size_fixed = 512
    raw_path = FLAGS.raw_path
    datepath = FLAGS.data_path
    if platform.uname()[1] != "dragonx-H97N-WIFI":
        raw_path = '/home/lxiaol9/ARC/EASTRNN/data/ICDAR/train/'
        datapath = "/home/lxiaol9/ARC/EASTRNN/data/GAP_process"
#     for dir_name, dir_names, file_names in os.walk(datapath):
#         for sub_dir_name in dir_names:
#             folder_list.append(sub_dir_name)
    cnt_video = 12
    video_set = ['Video_37_2_3', 'Video_51_7_4', 'Video_33_2_3', 'Video_47_6_4', 'Video_28_1_1', 'Video_40_2_3', 'Video_4_4_2', 'Video_42_2_3', 'Video_21_5_1', 'Video_54_7_4', 'Video_45_6_4', 'Video_16_3_2']
    print(video_set)
    frame_info = []
    video_name = []
    heat_map_collect = []
    geo_maps_collect = []
    score_maps_collect = []
    training_masks_collect = []
    for k in range(video_start-1, video_end):
        t1 = time.time()
        video_name.append(video_set[k])
        video_solo_path = datapath + '/' + video_set[k]
        # sort up all the paths
        xml_solo_path = raw_path+video_set[k]
        raw_video_path = raw_path + video_set[k]+'.mp4'
        cap = cv2.VideoCapture(raw_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # cnt_frame is frame number, heat_map_vect in (frame, w_d*h_d)
        heat_map_array, heat_map_vect, cnt_frame = read_heatmap(video_solo_path, w_d=size_fixed, h_d=size_fixed)
        # 1. load both polys and tags; 2. generate geo maps(the format of polys and tags need to match)
        polys_array_list, tags_array_list, id_list_list, frame_num = load_annotations_solo(xml_solo_path, 1, cnt_frame, frame_width, frame_height)
        geo_maps, score_maps, training_masks = single_video_geo(polys_array_list, tags_array_list, size_fixed)
        frame_info.append(cnt_frame)
        heat_map_collect.append(heat_map_array)
        geo_maps_collect.append(geo_maps)
        score_maps_collect.append(score_maps)
        training_masks_collect.append(training_masks)
        print('processed %d th video %s, taking %f seconds' %(k, video_set[k], time.time()-t1))
    #here we try to find a good data structure to store the list from each video
    return heat_map_collect, geo_maps_collect, score_maps_collect, training_masks_collect, frame_info, video_name


def frame_producer_mul(datapath, video_start,  video_end):
    # first know hwo many videos we have
    """
    :param datapath:
    :param video_start:
    :param video_end:
    :return: list of
    """
    folder_list = []
    size_fixed = 512
    raw_path = FLAGS.raw_path
    if platform.uname()[1] != "dragonx-H97N-WIFI":
        raw_path = '/home/lxiaol9/ARC/EASTRNN/data/ICDAR/train/'
#     for dir_name, dir_names, file_names in os.walk(datapath):
#         for sub_dir_name in dir_names:
#             folder_list.append(sub_dir_name)
    cnt_video = 25
    video_set = []
    for root, dirs, files in os.walk(raw_path):
        for file in files:
            if file.endswith('.mp4'):
                video_set.append(os.path.splitext(file)[0])
    print(video_set)
    frame_info = []
    video_name = []
    geo_maps_collect = []
    score_maps_collect = []
    training_masks_collect = []
    for k in range(video_start-1, video_end):
        t1 = time.time()
        video_name.append(video_set[k])
        video_solo_path = datapath + '/' + video_set[k]
        # sort up all the paths
        xml_solo_path = raw_path+video_set[k]
        raw_video_path = raw_path + video_set[k]+'.mp4'
        cap = cv2.VideoCapture(raw_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        cnt_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 1. load both polys and tags; 2. generate geo maps(the format of polys and tags need to match)
        polys_array_list, tags_array_list, id_list_list, frame_num = load_annotations_solo(xml_solo_path, 1, cnt_frame, frame_width, frame_height)
        geo_maps, score_maps, training_masks = single_video_geo(polys_array_list, tags_array_list, size_fixed)
        frame_info.append(cnt_frame)
        geo_maps_collect.append(geo_maps[:, ::4, ::4, :])
        score_maps_collect.append(score_maps[:, ::4, ::4])
        training_masks_collect.append(training_masks[:, ::4, ::4])
        print('processed %d th video %s, taking %f seconds' %(k, video_set[k], time.time()-t1))
    #here we try to find a good data structure to store the list from each video
    return geo_maps_collect, score_maps_collect, training_masks_collect, frame_info, video_name



def single_video_geo(polys_array, tags_set, img_size):
    """
    :param polys_array:
    :param tags_set:  IDs are needed here for tracking and recognition
    :param img_size:
    :return:
    """
    frame_num = len(polys_array)
    if FLAGS.geometry == "RBOX":
        geo_maps = np.zeros((frame_num, img_size, img_size, 5), dtype=np.float32)
    else:
        geo_maps = np.zeros((frame_num, img_size, img_size, 8), dtype=np.float32)
    score_maps = np.zeros((frame_num, img_size, img_size))
    training_masks = np.zeros((frame_num, img_size, img_size), dtype=np.int16)
    for index, polys_tags in enumerate(zip(polys_array, tags_set)):
        polys = polys_tags[0]
        tags = polys_tags[1]
        if any(t is None for t in [polys, tags]):
            continue
        if polys.shape[0] == 0:
            continue
        polys, tags = check_and_validate_polys(polys, tags, (img_size, img_size))
        geo_map, score_map, training_mask = geo_map_convert(polys, tags, img_size)
        geo_maps[index, :, :, :] = geo_map
        training_masks[index, :, :] = training_mask
        score_maps[index, :, :] = score_map

    return geo_maps, score_maps, training_masks


# Used for RBOX geometry maps generation
def geo_map_convert(polys, tags, img_size):
    """
    :param polys: normalized coordinates for text regions, in ()
    :param xmlpath: path to the gt file
    :param img_size: size of the input
    :return:
            1. the geo map either in RBOX(distance to four vertices, plus angle);
               or the QUAD(eight shifts in x,y axis)
            2. shrinked score map
            3. training mask with certain regions ignored
    Note: here it is in single frame, we have another function to create maps in video unit
          Also, the img_size is determined by the heat-map input and 2D LSTM output(we assume
          it's same as the hp input)
    """
    h = img_size
    w = img_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    training_mask = np.ones((h, w), dtype=np.uint8)
    if FLAGS.geometry == 'RBOX':
        # data format change happens before this function
        polys_list = polys
        tags_list = tags
        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]
            r = [None, None, None, None]
            for i in range(4):
                r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                           np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
            # score map
            shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
            cv2.fillPoly(score_map, shrinked_poly, 1)
            cv2.fillPoly(poly_mask, shrinked_poly, poly_idx + 1)
            # if the poly is too small, then ignore it during training
            poly_h = min(np.linalg.norm(poly[0] - poly[3]), np.linalg.norm(poly[1] - poly[2]))
            poly_w = min(np.linalg.norm(poly[0] - poly[1]), np.linalg.norm(poly[2] - poly[3]))
            if min(poly_h, poly_w) < FLAGS.min_text_size:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
            if tag:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

            xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
            # if geometry == 'RBOX':
            # 对任意两个顶点的组合生成一个平行四边形
            fitted_parallelograms = []
            for i in range(4):
                p0 = poly[i]
                p1 = poly[(i + 1) % 4]
                p2 = poly[(i + 2) % 4]
                p3 = poly[(i + 3) % 4]
                edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
                backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
                forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
                if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                    # 平行线经过p2
                    if edge[1] == 0:
                        edge_opposite = [1, 0, -p2[0]]
                    else:
                        edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
                else:
                    # 经过p3
                    if edge[1] == 0:
                        edge_opposite = [1, 0, -p3[0]]
                    else:
                        edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
                # move forward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p2 = line_cross_point(forward_edge, edge_opposite)
                if new_p2 is None:
                    print("new_p2 is None")
                    print(p0, p1, p2, p3)
                    new_p2 = p2
                if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                    # across p0
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p0[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
                else:
                    # across p3
                    if forward_edge[1] == 0:
                        forward_opposite = [1, 0, -p3[0]]
                    else:
                        forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
                new_p0 = line_cross_point(forward_opposite, edge)
                new_p3 = line_cross_point(forward_opposite, edge_opposite)
                if new_p0 is None:
                    print("new p0 is None")
                    print(p0, p1, p2, p3)
                    new_p0 = p0
                if new_p3 is None:
                    print("new p3 is None")
                    print(p0, p1, p2, p3)
                    new_p3 = p3
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
                # or move backward edge
                new_p0 = p0
                new_p1 = p1
                new_p2 = p2
                new_p3 = p3
                new_p3 = line_cross_point(backward_edge, edge_opposite)
                if new_p3 is None:
                    print("new p3 is None")
                    print(p0, p1, p2, p3)
                    new_p3 = p3
                if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                    # across p1
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p1[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
                else:
                    # across p2
                    if backward_edge[1] == 0:
                        backward_opposite = [1, 0, -p2[0]]
                    else:
                        backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
                new_p1 = line_cross_point(backward_opposite, edge)
                new_p2 = line_cross_point(backward_opposite, edge_opposite)
                if new_p1 is None:
                    print("new p1 is None")
                    print(p0, p1, p2, p3)
                    new_p1 = p1
                if new_p2 is None:
                    print("new p2 is None")
                    print(p0, p1, p2, p3)
                    new_p2 = p2
                fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            areas = [Polygon(t).area for t in fitted_parallelograms]
            parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
            # sort thie polygon
            parallelogram_coord_sum = np.sum(parallelogram, axis=1)
            min_coord_idx = np.argmin(parallelogram_coord_sum)
            parallelogram = parallelogram[
                [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

            rectange = rectangle_from_parallelogram(parallelogram)
            rectange, rotate_angle = sort_rectangle(rectange)

            p0_rect, p1_rect, p2_rect, p3_rect = rectange
            for y, x in xy_in_poly:
                point = np.array([x, y], dtype=np.float32)
                # top
                geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
                # right
                geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
                # down
                geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
                # left
                geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
                # angle
                geo_map[y, x, 4] = rotate_angle
        return geo_map, score_map, training_mask
    else:
        return None


if __name__ == '__main__':
    flags = tf.app.flags
    # The only part you need to modify during training
    flags.DEFINE_string("system", "local", "deciding running env")
    flags.DEFINE_string("data_path", "/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process", "Where data is stored")
    flags.DEFINE_integer("num_readers", 4, "process used to fetch data")
    FLAGS = flags.FLAGS

    datapath = '/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process'
    video_start = 1
    video_end = 2
    heat_map_collect, geo_maps_collect, score_maps_collect, training_masks_collect, frame_info, video_name = \
        array_producer_mul(datapath, video_start, video_end)
    print('memeda')
