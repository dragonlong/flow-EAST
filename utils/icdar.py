# coding:utf-8
import glob
import csv
import cv2
import time
import os
import numpy as np
import xmltodict
import random
from random import randint
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon
from shapely.geometry import MultiPoint

import tensorflow as tf
import sys
sys.path.append("../")
from utils.data_util import GeneratorEnqueuer
# from config.configrnn import get_config
# CONV  = "conv2d"
tf.app.flags.DEFINE_string('training_data_path', '/home/lxiaol9/ARC/EASTRNN/data/icdar2015/',
                           'training dataset to use')
tf.app.flags.DEFINE_integer('max_image_large_side', 1280,
                            'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800,
                            'if the text in the input image is bigger than this, then we resize'
                            'the image according to this')
tf.app.flags.DEFINE_integer('min_text_size', 5,
                            'if the text size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')
# add for test
# tf.app.flags.DEFINE_string('data_path', '/home/lxiaol9/ARC/EASTRNN/data/ICDAR/train', 'data of ICDAR')
#
# tf.app.flags.DEFINE_integer('input_size', 512, '')
#
# tf.app.flags.DEFINE_boolean('from_source', False, 'whether load data from source')
#
# tf.app.flags.DEFINE_string("model", "test", "A type of model. Possible options are: small, medium, large.")
#
# tf.app.flags.DEFINE_integer("num_gpus", 1, "Larger than 1 will create multiple training replicas")
#
# tf.app.flags.DEFINE_string("rnn_mode", CONV, "one of CUDNN: BASIC, BLOCK")

FLAGS = tf.app.flags.FLAGS

def get_images():
    files = []
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(FLAGS.training_data_path, '*.{}'.format(ext))))
    return files


def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            label = line[-1]
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]

            x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
            text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            if label == '*' or label == '###':
                text_tags.append(True)
            else:
                text_tags.append(False)
        return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)


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
                x1 = int(doc['Frames']['frame'][i - 1]['object']['Point'][0]['@x'])
                y1 = int(doc['Frames']['frame'][i - 1]['object']['Point'][0]['@y'])
                x2 = int(doc['Frames']['frame'][i - 1]['object']['Point'][1]['@x'])
                y2 = int(doc['Frames']['frame'][i - 1]['object']['Point'][1]['@y'])
                x3 = int(doc['Frames']['frame'][i - 1]['object']['Point'][2]['@x'])
                y3 = int(doc['Frames']['frame'][i - 1]['object']['Point'][2]['@y'])
                x4 = int(doc['Frames']['frame'][i - 1]['object']['Point'][3]['@x'])
                y4 = int(doc['Frames']['frame'][i - 1]['object']['Point'][3]['@y'])
                text_id = int(doc['Frames']['frame'][i-1]['object']['@ID'])
                text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                tags.append(False)
                id_container.append(text_id)
            else:
                object_num = len(doc['Frames']['frame'][i-1]['object'])
                for j in range(0, object_num):
                    x1 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@x'])
                    y1 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][0]['@y'])
                    x2 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@x'])
                    y2 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][1]['@y'])
                    x3 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@x'])
                    y3 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][2]['@y'])
                    x4 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@x'])
                    y4 = int(doc['Frames']['frame'][i-1]['object'][j]['Point'][3]['@y'])
                    text_id = int(doc['Frames']['frame'][i - 1]['object'][j]['@ID'])
                    text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                    tags.append(False)
                    id_container.append(text_id)
        polys_array_list.append(np.array(text_polys, dtype=np.float32))
        tags_array_list.append(np.array(tags, dtype=bool))
        id_list_list.append(np.array(id_container, dtype=np.int16))

    return polys_array_list, tags_array_list, id_list_list, frame_num

def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge)/2.


def check_and_validate_polys_old(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return polys, tags
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w-1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h-1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        poly_new = MultiPoint([tuple(poly[0]), tuple(poly[1]), tuple(poly[2]), tuple(poly[3])]).convex_hull
        if hasattr(poly_new, 'exterior'):
            x, y = poly_new.exterior.coords.xy
            poly[:, 0] = x[0:4]
            poly[:, 1] = y[0:4]
        else:
            x, y = poly_new.coords.xy
            poly = poly
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            # print poly
            # print('invalid poly')
            continue
        if p_area > 0:
            # print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    h, w, _ = im.shape
    pad_h = h//10
    pad_w = w//10
    h_array = np.zeros((h + pad_h*2), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx+pad_w:maxx+pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny+pad_h:maxy+pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w-1)
        xmax = np.clip(xmax, 0, w-1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h-1)
        ymax = np.clip(ymax, 0, h-1)
        if xmax - xmin < FLAGS.min_crop_side_ratio*w or ymax - ymin < FLAGS.min_crop_side_ratio*h:
            # area too small
            continue
        if polys.shape[0] != 0:
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax+1, xmin:xmax+1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


def crop_all_random(image, score_map, geo_map, tmask, crop_background=False, max_tries=20):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    # crop and assemble
    from random import randint
    flag = False # indication of success
    i_size = 512
    px_size = 512 # real patch size
    py_size = 512 # real patch size
    frame_height, frame_width = score_map.shape
    # return
    image_new = np.zeros((i_size, i_size, 3), dtype=np.uint8)
    score_new = np.zeros((i_size, i_size), dtype=np.uint8)
    geo_new = np.zeros((i_size, i_size, 5), dtype=np.float32)
    tmask_new = np.zeros((i_size, i_size), dtype=np.uint8)
    if frame_height < i_size:
        for i in range(max_tries*15):
            x = 0
            y = randint(2, frame_width-py_size-5)
            # print("Iterating over max_tries*15, {}".format(i))
            if (sum(sum(score_map[:, y-2:y+3])) + sum(sum(score_map[:, y+py_size-2:y+py_size+3]))) == 0:
                flag = True
                image_new[:frame_height, :py_size, :] = image[:, y:y+py_size, :]
                score_new[:frame_height, :py_size] = score_map[:, y:y+py_size]
                geo_new[:frame_height, :py_size, :]= geo_map[:, y:y+py_size, :]
                tmask_new[:frame_height, :py_size] = tmask[:, y:y+py_size]
                #print('Cropping succeed!!!')
                return image_new, score_new, geo_new, tmask_new
    else:
        new_h, new_w = frame_height, frame_width
        attempt_cnt = 0
        while attempt_cnt<max_tries:
            py_size = 512
            x = randint(2, new_h-px_size-5)
            y = randint(2, new_w-py_size-5)
            if score_map[x, y] > 0:
                continue
            attempt_cnt +=1
            # print('Iterating over py_size, {}'.format(attempt_cnt))
            for k in range(25):
                py_size-=10
                if (sum(sum(score_map[x:x+px_size, y-2:y+3])) + sum(sum(score_map[x:x+px_size, y+py_size-3:y+py_size+2])) + sum(sum(score_map[x-2:x+3, y:y+py_size])) + sum(sum(score_map[x+px_size-3:x+px_size+2, y:y+py_size]))) == 0:
                    # d = np.array([y, x, y, x+p_size-1,  y+p_size-1, x+p_size-1, y+p_size-1, x], dtype='int32')
                    # d = d.reshape(-1, 2)
                    flag = True
                    image_new[:px_size, :py_size, :] = image[x:x+px_size, y:y+py_size, :]
                    score_new[:px_size, :py_size] = score_map[x:x+px_size, y:y+py_size]
                    geo_new[:px_size, :py_size, :]= geo_map[x:x+px_size, y:y+py_size, :]
                    tmask_new[:px_size, :py_size] = tmask[x:x+px_size, y:y+py_size]
                    # new_img = score_map.copy()
                    # cv2.polylines(new_img, [d], isClosed=True, thickness=2, color=(1, 0, 0))
                    #print('Cropping succeed!!!')
                    return image_new, score_new, geo_new, tmask_new


    # print('Cropping failed, change the strategy!!!')
    return None, None, None, None

    # # visualization
    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(new_img)
    # plt.show()

def crop_all_random_seq(config, data, score_maps, geo_maps, training_masks, crop_background=False, max_tries=20):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    # crop and assemble
    score_map = score_maps[0, :, :]
    flag = False # indication of success
    px_size = 512 # real patch size
    py_size = 512 # real patch size
    frame_height, frame_width = score_map.shape
    # return
    images_new = np.zeros((config.num_steps, config.input_size, config.input_size, 3), dtype=np.uint8)
    scores_new = np.zeros((config.num_steps, config.input_size, config.input_size), dtype=np.uint8)
    geos_new = np.zeros((config.num_steps, config.input_size, config.input_size, 5), dtype=np.float32)
    tmasks_new = np.zeros((config.num_steps, config.input_size, config.input_size), dtype=np.uint8)
    if frame_height < config.input_size:
        for i in range(max_tries*15):
            x = 0
            y = randint(2, frame_width-py_size-5)
            # print("Iterating over max_tries*15, {}".format(i))
            if (sum(sum(score_map[:, y-2:y+3])) + sum(sum(score_map[:, y+py_size-2:y+py_size+3]))) == 0:
                flag = True
                images_new[:, :frame_height, :py_size, :] = data[:, :, y:y+py_size, :]
                scores_new[:, :frame_height, :py_size] = score_maps[:, :, y:y+py_size]
                geos_new[:, :frame_height, :py_size, :]= geo_maps[:, :, y:y+py_size, :]
                tmasks_new[:, :frame_height, :py_size] = training_masks[:, :, y:y+py_size]
                return images_new, scores_new, geos_new, tmasks_new
    else:
        new_h, new_w = frame_height, frame_width
        attempt_cnt = 0
        while attempt_cnt<max_tries:
            py_size = 512
            x = randint(2, new_h-px_size-5)
            y = randint(2, new_w-py_size-5)
            if score_map[x, y] > 0:
                continue
            attempt_cnt +=1
            # print('Iterating over py_size, {}'.format(attempt_cnt))
            for k in range(25):
                py_size-=10
                if (sum(sum(score_map[x:x+px_size, y-2:y+3])) + sum(sum(score_map[x:x+px_size, y+py_size-3:y+py_size+2])) + sum(sum(score_map[x-2:x+3, y:y+py_size])) + sum(sum(score_map[x+px_size-3:x+px_size+2, y:y+py_size]))) == 0:
                    # d = np.array([y, x, y, x+p_size-1,  y+p_size-1, x+p_size-1, y+p_size-1, x], dtype='int32')
                    # d = d.reshape(-1, 2)
                    flag = True
                    images_new[:, :px_size, :py_size, :] = data[:, x:x+px_size, y:y+py_size, :]
                    scores_new[:, :px_size, :py_size] = score_maps[:, x:x+px_size, y:y+py_size]
                    geos_new[:, :px_size, :py_size, :]= geo_maps[:, x:x+px_size, y:y+py_size, :]
                    tmasks_new[:, :px_size, :py_size] = training_masks[:, x:x+px_size, y:y+py_size]
                    # new_img = score_map.copy()
                    # cv2.polylines(new_img, [d], isClosed=True, thickness=2, color=(1, 0, 0))
                    #print('Cropping succeed!!!')
                    return images_new, scores_new, geos_new, tmasks_new


    # print('Cropping failed, change the strategy!!!')
    return None, None, None, None

    # # visualization
    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(new_img)
    # plt.show()

def crop_center_random_seq(config, data, score_map, geo_map, training_mask, crop_flag = True, crop_background=False, max_tries=20):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    '''
    # crop and assemble
    flag = False # indication of success
    px_size = 512 # real patch size
    py_size = 512 # real patch size
    frame_height, frame_width = score_map.shape
    # return
    images_new = np.zeros((config.num_steps, config.input_size, config.input_size, 3), dtype=np.uint8)
    scores_new = np.zeros((config.input_size, config.input_size), dtype=np.uint8)
    geos_new = np.zeros((config.input_size, config.input_size, 5), dtype=np.float32)
    tmasks_new = np.zeros((config.input_size, config.input_size), dtype=np.uint8)
    if not crop_flag :
        if frame_height < config.input_size:
            y = 5
            images_new[:, :frame_height, :py_size, :] = data[:, :, y:y+py_size, :]
            scores_new[:frame_height, :py_size] = score_map[:, y:y+py_size]
            geos_new[:frame_height, :py_size, :]= geo_map[:, y:y+py_size, :]
            tmasks_new[:frame_height, :py_size] = training_mask[:, y:y+py_size]
        else:
            x = 5
            y = 5
            images_new[:, :px_size, :py_size, :] = data[:, x:x+px_size, y:y+py_size, :]
            scores_new[:px_size, :py_size] = score_map[x:x+px_size, y:y+py_size]
            geos_new[:px_size, :py_size, :]= geo_map[x:x+px_size, y:y+py_size, :]
            tmasks_new[:px_size, :py_size] = training_mask[x:x+px_size, y:y+py_size]
        return images_new, scores_new, geos_new, tmasks_new
    if frame_height < config.input_size:
        for i in range(max_tries*15):
            x = 0
            y = randint(2, frame_width-py_size-5)
            # print("Iterating over max_tries*15, {}".format(i))
            if (sum(sum(score_map[:, y-2:y+3])) + sum(sum(score_map[:, y+py_size-2:y+py_size+3]))) == 0:
                flag = True
                images_new[:, :frame_height, :py_size, :] = data[:, :, y:y+py_size, :]
                scores_new[:frame_height, :py_size] = score_map[:, y:y+py_size]
                geos_new[:frame_height, :py_size, :]= geo_map[:, y:y+py_size, :]
                tmasks_new[:frame_height, :py_size] = training_mask[:, y:y+py_size]
                return images_new, scores_new, geos_new, tmasks_new
    else:
        new_h, new_w = frame_height, frame_width
        attempt_cnt = 0
        while attempt_cnt<max_tries:
            py_size = 512
            x = randint(2, new_h-px_size-5)
            y = randint(2, new_w-py_size-5)
            if score_map[x, y] > 0:
                continue
            attempt_cnt +=1
            # print('Iterating over py_size, {}'.format(attempt_cnt))
            for k in range(25):
                py_size-=10
                if (sum(sum(score_map[x:x+px_size, y-2:y+3])) + sum(sum(score_map[x:x+px_size, y+py_size-3:y+py_size+2])) + sum(sum(score_map[x-2:x+3, y:y+py_size])) + sum(sum(score_map[x+px_size-3:x+px_size+2, y:y+py_size]))) == 0:
                    flag = True
                    images_new[:, :px_size, :py_size, :] = data[:, x:x+px_size, y:y+py_size, :]
                    scores_new[:px_size, :py_size] = score_map[x:x+px_size, y:y+py_size]
                    geos_new[:px_size, :py_size, :]= geo_map[x:x+px_size, y:y+py_size, :]
                    tmasks_new[:px_size, :py_size] = training_mask[x:x+px_size, y:y+py_size]
                    # new_img = score_map.copy()
                    return images_new, scores_new, geos_new, tmasks_new


    # print('Cropping failed, change the strategy!!!')
    return None, None, None, None

    # # visualization
    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(new_img)
    # plt.show()


def shrink_poly(poly, r):
    '''
    fit a poly inside the origin poly, maybe bugs here...
    used for generate the score map
    :param poly: the text poly
    :param r: r in the paper
    :return: the shrinked poly
    '''
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                    np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        ## p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        ## p0, p3
        # print poly
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        ## p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        ## p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        ## p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    if any(t is None for t in [p1, p2, p3]):
        #print(p1, p2, p3)
        return 0
    elif np.linalg.norm(p2 - p1)==0:
        return 0
    else:
        return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        # print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        # print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1-b2)/(k1-k2)
        y = k1*x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1./line[0], -1, point[1] - (-1/line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1-p0, p3-p0)/(np.linalg.norm(p0-p1) * np.linalg.norm(p3-p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0-p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0-p1) > np.linalg.norm(p0-p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # 底边平行于X轴, 那么p0为左上角
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # 找到最低点右边的点
        p_lowest_right = (p_lowest - 1) % 4
        p_lowest_left = (p_lowest + 1) % 4
        angle = np.arctan(-(poly[p_lowest][1] - poly[p_lowest_right][1])/(poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle/np.pi * 180 > 45:
            # 这个点为p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            # 这个点为p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)


def generate_rbox_single(rbox):
    poly = rbox.reshape([4,2])
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
            # print("new_p2 is None")
            # print(p0, p1, p2, p3)
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
            # print("new p0 is None")
            # print(p0, p1, p2, p3)
            new_p0 = p0
        if new_p3 is None:
            # print("new p3 is None")
            # print(p0, p1, p2, p3)
            new_p3 = p3
        fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        # or move backward edge
        new_p0 = p0
        new_p1 = p1
        new_p2 = p2
        new_p3 = p3
        new_p3 = line_cross_point(backward_edge, edge_opposite)
        if new_p3 is None:
            # print("new p3 is None")
            # print(p0, p1, p2, p3)
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
            # print("new p1 is None")
            # print(p0, p1, p2, p3)
            new_p1 = p1
        if new_p2 is None:
            # print("new p2 is None")
            # print(p0, p1, p2, p3)
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
    # actually this reshaped bbox is a kind of attention
    p0_rect, p1_rect, p2_rect, p3_rect = rectange
    #
    p_center = p0_rect+ p1_rect+ p2_rect+ p3_rect
    x, y = p_center/4
    point = np.array([x, y], dtype=np.float32)
    # top
    d1 = point_dist_to_line(p0_rect, p1_rect, point)
    # right
    d2 = point_dist_to_line(p1_rect, p2_rect, point)
    # down
    d3 = point_dist_to_line(p2_rect, p3_rect, point)
    # left
    d4 = point_dist_to_line(p3_rect, p0_rect, point)
    # x, is on horizontal direction, y is on vertical direction
    return np.array([x, y, d1, d2, rotate_angle])


def generate_rbox(im_size, polys, tags):
    h, w = im_size
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    score_map = np.zeros((h, w), dtype=np.uint8)
    geo_map = np.zeros((h, w, 5), dtype=np.float32)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
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
                # print("new_p2 is None")
                # print(p0, p1, p2, p3)
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
                # print("new p0 is None")
                # print(p0, p1, p2, p3)
                new_p0 = p0
            if new_p3 is None:
                # print("new p3 is None")
                # print(p0, p1, p2, p3)
                new_p3 = p3
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if new_p3 is None:
                # print("new p3 is None")
                # print(p0, p1, p2, p3)
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
                # print("new p1 is None")
                # print(p0, p1, p2, p3)
                new_p1 = p1
            if new_p2 is None:
                # print("new p2 is None")
                # print(p0, p1, p2, p3)
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
    return score_map, geo_map, training_mask


def generator(input_size=512, batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 2.0, 3.0]),
              vis=False):
    image_list = np.array(get_images())
    print('{} training images in {}'.format(
        image_list.shape[0], FLAGS.training_data_path))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        training_masks = []
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                # print im_fn
                h, w, _ = im.shape
                txt_fn = im_fn.replace(os.path.basename(im_fn).split('.')[1], 'txt')
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue

                text_polys, text_tags = load_annoataion(txt_fn)

                text_polys, text_tags = check_and_validate_polys_old(text_polys, text_tags, (h, w))
                # if text_polys.shape[0] == 0:
                #     continue
                # random scale this image
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale
                # print rd_scale
                # random crop a area from image
                if np.random.rand() < background_ratio:
                    # crop background
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                    if text_polys.shape[0] > 0:
                        # cannot find background
                        continue
                    # pad and resize image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size))
                    score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                    geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                else:
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                    if text_polys.shape[0] == 0:
                        continue
                    h, w, _ = im.shape

                    # pad the image to the training input size or the longer side of image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = im_padded
                    # resize the image to input size
                    new_h, new_w, _ = im.shape
                    resize_h = input_size
                    resize_w = input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    resize_ratio_3_x = resize_w/float(new_w)
                    resize_ratio_3_y = resize_h/float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape
                    score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)

                if vis:
                    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                    # axs[0].imshow(im[:, :, ::-1])
                    # axs[0].set_xticks([])
                    # axs[0].set_yticks([])
                    # for poly in text_polys:
                    #     poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                    #     poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                    #     axs[0].add_artist(Patches.Polygon(
                    #         poly * 4, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                    #     axs[0].text(poly[0, 0] * 4, poly[0, 1] * 4, '{:.0f}-{:.0f}'.format(poly_h * 4, poly_w * 4),
                    #                    color='purple')
                    # axs[1].imshow(score_map)
                    # axs[1].set_xticks([])
                    # axs[1].set_yticks([])
                    axs[0, 0].imshow(im[:, :, ::-1])
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    for poly in text_polys:
                        poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                        poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                        axs[0, 0].add_artist(Patches.Polygon(
                            poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                        axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                    axs[0, 1].imshow(score_map[::, ::])
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[1, 0].imshow(geo_map[::, ::, 0])
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    axs[1, 1].imshow(geo_map[::, ::, 1])
                    axs[1, 1].set_xticks([])
                    axs[1, 1].set_yticks([])
                    axs[2, 0].imshow(geo_map[::, ::, 2])
                    axs[2, 0].set_xticks([])
                    axs[2, 0].set_yticks([])
                    axs[2, 1].imshow(training_mask[::, ::])
                    axs[2, 1].set_xticks([])
                    axs[2, 1].set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                images.append(im[:, :, ::-1].astype(np.float32))
                image_fns.append(im_fn)
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    yield images, image_fns, score_maps, geo_maps, training_masks
                    images = []
                    image_fns = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def generator_fast(input_size=512, batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 1.5, 2.0]),
              vis=False):
    # for one round, we will randomly choose video, and shuffe frame index, then apply crop_poly and rbox generation
    video_set = []
    for root, dirs, files in os.walk(FLAGS.data_path):
        for file in files:
            if file.endswith('.mp4'):
                video_set.append(os.path.splitext(file)[0])
    print(video_set)
    # video_set = ['Video_37_2_3', 'Video_51_7_4', 'Video_33_2_3', 'Video_47_6_4', 'Video_28_1_1', 'Video_40_2_3', \
    #              'Video_4_4_2', 'Video_42_2_3', 'Video_21_5_1', 'Video_54_7_4', 'Video_45_6_4', 'Video_16_3_2']
    index = np.arange(0, len(video_set))
    images_array_list = []
    score_maps_array_list = []
    geo_maps_array_list = []
    training_masks_array_list = []
    frame_info = []
    polys_array_list2 = []
    tags_array_list2 = []
    for k in index:
        t_start = time.time()
        # sort up all the paths
        xml_solo_path = FLAGS.data_path + '/' + video_set[k]
        raw_video_path = FLAGS.data_path + '/' + video_set[k]+'.mp4'
        cap = cv2.VideoCapture(raw_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        cnt_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_info.append(cnt_frame)
        # 1. load both polys and tags; 2. generate geo maps(the format of polys and tags need to match)
        polys_array_list, tags_array_list, id_list_list, frame_num = load_annotations_solo(xml_solo_path, 1, cnt_frame, frame_width, frame_height)
        polys_array_list2.append(polys_array_list)
        tags_array_list2.append(tags_array_list)
        # 2. apply RBOX box generation on the whole frame
        # images_array = np.zeros((cnt_frame, frame_height, frame_width, 3), dtype=np.float32)
        # score_maps_array = np.zeros((cnt_frame, frame_height, frame_width), dtype=np.uint8)
        # geo_maps_array = np.zeros((cnt_frame, frame_height, frame_width, 5), dtype=np.float32)
        # training_masks_array = np.zeros((cnt_frame, frame_height, frame_width), dtype=np.uint8)
        if FLAGS.from_source is True:
            for m in range(cnt_frame):
                ret, image = cap.read()
                if ret is True:
                    images_array[m, :, :, :] = image
                else:
                    print('Hey, we got something wrong here!')
                    continue
                # text_polys, text_tags = load_annoataion(txt_fn)
                text_polys, text_tags = polys_array_list[m], tags_array_list[m]
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (frame_height, frame_width))
                    # im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                if text_polys.shape[0] == 0:
                    continue
                new_h, new_w = frame_height, frame_width
                score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
                score_maps_array[m, :, :] = score_map
                geo_maps_array[m, :, :, :] = geo_map
                training_masks_array[m, :, :] = training_mask

            basename = '/work/cascades/lxiaol9/ARC/EAST/data/pre-processed'
            img_name = basename + '/images/' + '{}'.format(video_set[k])+'.npy'
            score_name = basename + '/score_maps/' + '{}'.format(video_set[k])+'.npy'
            geo_name = basename + '/geo_maps/' + '{}'.format(video_set[k])+'.npy'
            mask_name = basename + '/training_masks/' + '{}'.format(video_set[k])+'.npy'
            np.save(img_name, images_array)
            np.save(score_name, score_maps_array)
            np.save(geo_name, geo_maps_array)
            np.save(mask_name, training_masks_array)
            #print('Video {} has size ({}, {}, {}) '.format(video_set[k], frame_width, frame_height, cnt_frame))
            print('The preprocessing for Video {} has cost {} seconds'.format(video_set[k], time.time() - t_start))
        else:
            # basename = '/work/cascades/lxiaol9/ARC/EAST/data/pre-processed'
            # img_name = basename + '/images/' + '{}'.format(video_set[k])+'.npy'
            # score_name = basename + '/score_maps/' + '{}'.format(video_set[k])+'.npy'
            # geo_name = basename + '/geo_maps/' + '{}'.format(video_set[k])+'.npy'
            # mask_name = basename + '/training_masks/' + '{}'.format(video_set[k])+'.npy'
            # score_maps_array_list.append(np.load(score_name))
            # geo_maps_array_list.append(np.load(geo_name))
            # training_masks_array_list.append(np.load(mask_name))
            print('The preprocessing for Video {} has cost {} seconds'.format(video_set[k], time.time() - t_start))
    # Now we just need to randomly cut and apply scale to both the image and ground-truth
    # we have the image_array_list, score_maps_array_list, geo_maps_array_list, training_masks_array_list
    # ctrl_cnt = 100
    # for t in range(ctrl_cnt):
    while True:
        index1 = index
        np.random.shuffle(index1)
        index2 = index1
        np.random.shuffle(index2)
        index3 = index2
        np.random.shuffle(index3)
        index_double = np.concatenate((index1, index2, index3), axis=0)
        images = []
        score_maps = []
        geo_maps = []
        training_masks = []
        for i in index_double:
            try:
                j = randint(0, frame_info[i]-1)
                video_path = FLAGS.data_path + '/' + video_set[i]+'.mp4'
                cap = cv2.VideoCapture(video_path)
                cap.set(1, j)
                ret, image = cap.read()
                ######### add on-line data processing, Sep 5th by Xiaolong
                text_polys, text_tags = polys_array_list2[i][j], tags_array_list2[i][j]
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (frame_height, frame_width))
                if text_polys.shape[0] == 0:
                    continue
                new_h, new_w = int(cap.get(4)), int(cap.get(3))
                score_map_raw, geo_map_raw, training_mask_raw = generate_rbox((new_h, new_w), text_polys, text_tags)
                ######### end
                #print("Cropping images from {} !!!".format(video_set[i]))
                img, score_map, geo_map, training_mask = crop_all_random(image, score_map_raw, geo_map_raw, training_mask_raw)
                # img, score_map, geo_map, training_mask = crop_all_random(image, score_maps_array_list[i][j, :, :], \
                #             geo_maps_array_list[i][j, :, :, :], training_masks_array_list[i][j, :, :])
                if img is not None:
                    images.append(img.astype(np.float32))
                    score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                    geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                    training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))
                if len(images) == batch_size:
                    yield images, score_maps, geo_maps, training_masks
                    images = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

# used for
def generator_fast_sequence(config=None, is_training=True,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 1.5, 2.0]),
              vis=False):
    video_set = []
    for root, dirs, files in os.walk(FLAGS.data_path):
        for file in files:
            if file.endswith('.mp4'):
                video_set.append(os.path.splitext(file)[0])
    print(video_set)
    # video_set = ['Video_37_2_3', 'Video_51_7_4', 'Video_33_2_3', 'Video_47_6_4', 'Video_28_1_1', 'Video_40_2_3', \
    #              'Video_4_4_2', 'Video_42_2_3', 'Video_21_5_1', 'Video_54_7_4', 'Video_45_6_4', 'Video_16_3_2']
    if is_training:
        index = np.arange(0, len(video_set)-3)
    else:
        index = np.arange(0, 3)
        video_set = video_set[(len(video_set)-3):]
    images_array_list = []
    score_maps_array_list = []
    geo_maps_array_list = []
    training_masks_array_list = []
    frame_info = []
    polys_array_list2 = []
    tags_array_list2 = []
    for k in index:
        t_start = time.time()
        # sort up all the paths
        xml_solo_path = FLAGS.data_path + '/' + video_set[k]
        raw_video_path = FLAGS.data_path + '/' + video_set[k]+'.mp4'
        cap = cv2.VideoCapture(raw_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        cnt_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_info.append(cnt_frame)
        # 1. load both polys and tags; 2. generate geo maps(the format of polys and tags need to match)
        polys_array_list, tags_array_list, id_list_list, frame_num = load_annotations_solo(xml_solo_path, 1, cnt_frame, frame_width, frame_height)
        polys_array_list2.append(polys_array_list)
        tags_array_list2.append(tags_array_list)
        # 2. apply RBOX box generation on the whole frame
        # images_array = np.zeros((cnt_frame, frame_height, frame_width, 3), dtype=np.float32)
        score_maps_array = np.zeros((cnt_frame, frame_height, frame_width), dtype=np.uint8)
        geo_maps_array = np.zeros((cnt_frame, frame_height, frame_width, 5), dtype=np.float32)
        training_masks_array = np.zeros((cnt_frame, frame_height, frame_width), dtype=np.uint8)
        if FLAGS.from_source is True:
            for m in range(cnt_frame):
                ret, image = cap.read()
                if ret is True:
                    None
                    #images_array[m, :, :, :] = image
                else:
                    print('Hey, we got something wrong here!')
                    continue
                # text_polys, text_tags = load_annoataion(txt_fn)
                text_polys, text_tags = polys_array_list[m], tags_array_list[m]
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (frame_height, frame_width))
                    # im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                if text_polys.shape[0] == 0:
                    continue
                new_h, new_w = frame_height, frame_width
                score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
                score_maps_array[m, :, :] = score_map
                geo_maps_array[m, :, :, :] = geo_map
                training_masks_array[m, :, :] = training_mask

            basename = '/work/cascades/lxiaol9/ARC/EAST/data/pre-processed'
            # img_name = basename + '/images/' + '{}'.format(video_set[k])+'.npy'
            # score_name = basename + '/score_maps/' + '{}'.format(video_set[k])+'.npy'
            # geo_name = basename + '/geo_maps/' + '{}'.format(video_set[k])+'.npy'
            # mask_name = basename + '/training_masks/' + '{}'.format(video_set[k])+'.npy'
            # np.save(img_name, images_array)
            # np.save(score_name, score_maps_array)
            # np.save(geo_name, geo_maps_array)
            # np.save(mask_name, training_masks_array)
            #print('Video {} has size ({}, {}, {}) '.format(video_set[k], frame_width, frame_height, cnt_frame))
            print('The preprocessing for Video {} has cost {} seconds'.format(video_set[k], time.time() - t_start))
        # else:
            # basename = '/work/cascades/lxiaol9/ARC/EAST/data/pre-processed'
            # img_name = basename + '/images/' + '{}'.format(video_set[k])+'.npy'
            # score_name = basename + '/score_maps/' + '{}'.format(video_set[k])+'.npy'
            # geo_name = basename + '/geo_maps/' + '{}'.format(video_set[k])+'.npy'
            # mask_name = basename + '/training_masks/' + '{}'.format(video_set[k])+'.npy'
            # score_maps_array_list.append(np.load(score_name))
            # geo_maps_array_list.append(np.load(geo_name))
            # training_masks_array_list.append(np.load(mask_name))
            # print('The preprocessing for Video {} has cost {} seconds'.format(video_set[k], time.time() - t_start))
    # ctrl_cnt = 100
    # for t in range(ctrl_cnt):
    while True:
        index1 = index
        np.random.shuffle(index1)
        index2 = index1
        np.random.shuffle(index2)
        index3 = index2
        np.random.shuffle(index3)
        index_double = np.concatenate((index1, index2, index3), axis=0)
        images_seq = []
        score_maps_seq = []
        geo_maps_seq = []
        training_masks_seq = []
        for i in index_double:
            try:
                video_path = FLAGS.data_path + '/' + video_set[i]+'.mp4'
                cap = cv2.VideoCapture(video_path)
                new_h, new_w = int(cap.get(4)), int(cap.get(3))
                data = np.zeros([config.num_steps, new_h, new_w, 3], dtype=np.uint8)
                score_maps = np.zeros([config.num_steps, new_h, new_w], dtype=np.uint8)
                geo_maps = np.zeros([config.num_steps, new_h, new_w, 5], dtype=np.float32)
                training_masks = np.zeros([config.num_steps, new_h, new_w], dtype=np.uint8)
                j = randint(0, frame_info[i]-config.num_steps)
                for m in range(config.num_steps):
                    cap.set(1, j+m)
                    ret, image = cap.read()
                    ######### add on-line data processing, Sep 5th by Xiaolong
                    text_polys, text_tags = polys_array_list2[i][j+m], tags_array_list2[i][j+m]
                    text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (frame_height, frame_width))
                    if text_polys.shape[0] == 0:
                        data[m, :, :, :] = image
                        continue
                    score_map_raw, geo_map_raw, training_mask_raw = generate_rbox((new_h, new_w), text_polys, text_tags)
                    data[m, :, :, :] = image
                    score_maps[m, :, :] = score_map_raw
                    geo_maps[m, :, :, :] = geo_map_raw
                    training_masks[m, :, :] = training_mask_raw
                imgs_c, score_maps_c, geo_maps_c, training_masks_c = crop_all_random_seq(config, data, score_maps, geo_maps, training_masks)
                print("Cropping one video sequence")
                if imgs_c is not None:
                    images_seq.append(imgs_c.astype(np.float32))
                    score_maps_seq.append(score_maps_c[:, ::4, ::4, np.newaxis].astype(np.float32))
                    geo_maps_seq.append(geo_maps_c[:, ::4, ::4, :].astype(np.float32))
                    training_masks_seq.append(training_masks_c[:, ::4, ::4, np.newaxis].astype(np.float32))
                if len(images_seq) == config.batch_size:
                    yield images_seq, score_maps_seq, geo_maps_seq, training_masks_seq
                    images_seq = []
                    score_maps_seq = []
                    geo_maps_seq = []
                    training_masks_seq = []
                    # return images_seq, score_maps_seq, geo_maps_seq, training_masks_seq
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def generator_center_seq(config=None, is_training=True,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 1.5, 2.0]),
              vis=False):
    video_set = []
    for root, dirs, files in os.walk(FLAGS.data_path):
        for file in files:
            if file.endswith('.mp4'):
                video_set.append(os.path.splitext(file)[0])
    if is_training:
        index = np.arange(0, len(video_set))
    else:
        index = np.arange(0, 3)
        video_set = video_set[(len(video_set)-3):]
    images_array_list = []
    score_maps_array_list = []
    geo_maps_array_list = []
    training_masks_array_list = []
    frame_info = []
    polys_array_list2 = []
    tags_array_list2 = []
    for k in index:
        t_start = time.time()
        # sort up all the paths
        xml_solo_path = FLAGS.data_path + '/' + video_set[k]
        raw_video_path = FLAGS.data_path + '/' + video_set[k]+'.mp4'
        cap = cv2.VideoCapture(raw_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        cnt_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_info.append(cnt_frame)
        # 1. load both polys and tags; 2. generate geo maps(the format of polys and tags need to match)
        polys_array_list, tags_array_list, id_list_list, frame_num = load_annotations_solo(xml_solo_path, 1, cnt_frame, frame_width, frame_height)
        polys_array_list2.append(polys_array_list)
        tags_array_list2.append(tags_array_list)
    # ctrl_cnt = 100
    # for t in range(ctrl_cnt):
    while True:
        index1 = index
        np.random.shuffle(index1)
        index2 = index1
        np.random.shuffle(index2)
        index3 = index2
        np.random.shuffle(index3)
        index_double = np.concatenate((index1, index2, index3), axis=0)
        images_seq = []
        score_maps_seq = []
        geo_maps_seq = []
        training_masks_seq = []
        crop_flag = False
        for i in index_double:
            try:
                video_path = FLAGS.data_path + '/' + video_set[i]+'.mp4'
                cap = cv2.VideoCapture(video_path)
                new_h, new_w = int(cap.get(4)), int(cap.get(3))
                data = np.zeros([config.num_steps, new_h, new_w, 3], dtype=np.uint8)
                score_maps = np.zeros([new_h, new_w], dtype=np.uint8)
                geo_maps = np.zeros([new_h, new_w, 5], dtype=np.float32)
                training_masks = np.zeros([new_h, new_w], dtype=np.uint8)
                # j range()
                j = randint(config.sample_range, frame_info[i]-config.sample_range)
                lower_bound = j - config.sample_range
                upper_bound = j + config.sample_range
                prev_frames = np.sort(random.sample(range(lower_bound, j), 2))
                post_frames = np.sort(random.sample(range(j+1, upper_bound), 2))
                index_lookup = [prev_frames[0], prev_frames[1], j, post_frames[0], post_frames[1]]
                # num of steps,
                # index_lookup = [j-1, j, j+1]
                for m, ind in enumerate(index_lookup):
                    cap.set(1, ind)
                    ret, image = cap.read()
                    ######### add on-line data processing, Sep 5th by Xiaolong
                    text_polys, text_tags = polys_array_list2[i][ind], tags_array_list2[i][ind]
                    text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (frame_height, frame_width))
                    if text_polys.shape[0] == 0:
                        data[m, :, :, :] = image
                        continue
                    if m == 2:
                        score_maps, geo_maps, training_masks = generate_rbox((new_h, new_w), text_polys, text_tags)
                        crop_flag = True
                    data[m, :, :, :] = image

                imgs_c, score_maps_c, geo_maps_c, training_masks_c = crop_center_random_seq(config, data, score_maps, geo_maps, training_masks, crop_flag)
                crop_flag = False
                if imgs_c is not None:
                    images_seq.append(imgs_c.astype(np.float32))
                    score_maps_seq.append(score_maps_c[::4, ::4, np.newaxis].astype(np.float32))
                    geo_maps_seq.append(geo_maps_c[::4, ::4, :].astype(np.float32))
                    training_masks_seq.append(training_masks_c[::4, ::4, np.newaxis].astype(np.float32))
                if len(images_seq) == FLAGS.batch_size_per_gpu*FLAGS.num_gpus:
                    yield images_seq, score_maps_seq, geo_maps_seq, training_masks_seq
                    images_seq = []
                    score_maps_seq = []
                    geo_maps_seq = []
                    training_masks_seq = []
                    # return images_seq, score_maps_seq, geo_maps_seq, training_masks_seq
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def generator_for_video(input_size=512, batch_size=32,
              background_ratio=3./8,
              random_scale=np.array([0.5, 1, 1.5, 2.0]),
              vis=False):
    # for one round, we will randomly choose video, and shuffe frame index, then apply crop_poly and rbox generation
    # image_list = np.array(get_images())
    # print('{} training images in {}'.format(

    #     image_list.shape[0], FLAGS.training_data_path))
    video_set = ['Video_37_2_3', 'Video_51_7_4', 'Video_33_2_3', 'Video_47_6_4', 'Video_28_1_1', 'Video_40_2_3', \
                 'Video_4_4_2', 'Video_42_2_3', 'Video_21_5_1', 'Video_54_7_4', 'Video_45_6_4', 'Video_16_3_2']
    # for root, dirs, files in os.walk(FLAGS.data_path):
    #     for file in files:
    #         if file.endswith('.mp4'):
    #             video_set.append(os.path.splitext(file)[0])
    index = np.arange(0, len(video_set))
    images_array_list = []
    score_maps_array_list = []
    geo_maps_array_list = []
    training_masks_array_list = []
    frame_info = []
    polys_array_list2 = []
    tags_array_list2 = []
    for k in index:
        t_start = time.time()
        print("start to process video "+video_set[k])
        # sort up all the paths
        xml_solo_path = FLAGS.data_path + '/' + video_set[k]
        raw_video_path = FLAGS.data_path + '/' + video_set[k]+'.mp4'
        cap = cv2.VideoCapture(raw_video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        cnt_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_info.append(cnt_frame)
        # 1. load both polys and tags; 2. generate geo maps(the format of polys and tags need to match)
        polys_array_list, tags_array_list, id_list_list, frame_num = load_annotations_solo(xml_solo_path, 1, cnt_frame, frame_width, frame_height)
        polys_array_list2.append(polys_array_list)
        tags_array_list2.append(tags_array_list)
        # 2. apply RBOX box generation on the whole frame
        # images_array = np.zeros((cnt_frame, frame_height, frame_width, 3), dtype=np.float32)
        # score_maps_array = np.zeros((cnt_frame, frame_height, frame_width), dtype=np.uint8)
        # geo_maps_array = np.zeros((cnt_frame, frame_height, frame_width, 5), dtype=np.float32)
        # training_masks_array = np.zeros((cnt_frame, frame_height, frame_width), dtype=np.uint8)
        if FLAGS.from_source is True:
            for m in range(cnt_frame):
                ret, image = cap.read()
                if ret is True:
                    images_array[m, :, :, :] = image
                else:
                    print('Hey, we got something wrong here!')
                    continue
                # text_polys, text_tags = load_annoataion(txt_fn)
                text_polys, text_tags = polys_array_list[m], tags_array_list[m]
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (frame_height, frame_width))
                    # im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                if text_polys.shape[0] == 0:
                    continue
                new_h, new_w = frame_height, frame_width
                score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
                score_maps_array[m, :, :] = score_map
                geo_maps_array[m, :, :, :] = geo_map
                training_masks_array[m, :, :] = training_mask

                if vis:
                    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                    # axs[0].imshow(im[:, :, ::-1])
                    # axs[0].set_xticks([])
                    # axs[0].set_yticks([])
                    # for poly in text_polys:
                    #     poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                    #     poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                    #     axs[0].add_artist(Patches.Polygon(
                    #         poly * 4, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                    #     axs[0].text(poly[0, 0] * 4, poly[0, 1] * 4, '{:.0f}-{:.0f}'.format(poly_h * 4, poly_w * 4),
                    #                    color='purple')
                    # axs[1].imshow(score_map)
                    # axs[1].set_xticks([])
                    # axs[1].set_yticks([])
                    axs[0, 0].imshow(im[:, :, ::-1])
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    for poly in text_polys:
                        poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                        poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                        axs[0, 0].add_artist(Patches.Polygon(
                            poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                        axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                    axs[0, 1].imshow(score_map[::, ::])
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[1, 0].imshow(geo_map[::, ::, 0])
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    axs[1, 1].imshow(geo_map[::, ::, 1])
                    axs[1, 1].set_xticks([])
                    axs[1, 1].set_yticks([])
                    axs[2, 0].imshow(geo_map[::, ::, 2])
                    axs[2, 0].set_xticks([])
                    axs[2, 0].set_yticks([])
                    axs[2, 1].imshow(training_mask[::, ::])
                    axs[2, 1].set_xticks([])
                    axs[2, 1].set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()
            basename = '/work/cascades/lxiaol9/ARC/EAST/data/pre-processed'
            img_name = basename + '/images/' + '{}'.format(video_set[k])+'.npy'
            score_name = basename + '/score_maps/' + '{}'.format(video_set[k])+'.npy'
            geo_name = basename + '/geo_maps/' + '{}'.format(video_set[k])+'.npy'
            mask_name = basename + '/training_masks/' + '{}'.format(video_set[k])+'.npy'
            np.save(img_name, images_array)
            np.save(score_name, score_maps_array)
            np.save(geo_name, geo_maps_array)
            np.save(mask_name, training_masks_array)
            #print('Video {} has size ({}, {}, {}) '.format(video_set[k], frame_width, frame_height, cnt_frame))
            print('The preprocessing for Video {} has cost {} seconds'.format(video_set[k], time.time() - t_start))
        else:
            # basename = '/work/cascades/lxiaol9/ARC/EAST/data/pre-processed'
            # img_name = basename + '/images/' + '{}'.format(video_set[k])+'.npy'
            # score_name = basename + '/score_maps/' + '{}'.format(video_set[k])+'.npy'
            # geo_name = basename + '/geo_maps/' + '{}'.format(video_set[k])+'.npy'
            # mask_name = basename + '/training_masks/' + '{}'.format(video_set[k])+'.npy'
            # score_maps_array_list.append(np.load(score_name))
            # geo_maps_array_list.append(np.load(geo_name))
            # training_masks_array_list.append(np.load(mask_name))
            print('The preprocessing for Video {} has cost {} seconds'.format(video_set[k], time.time() - t_start))

    while True:
        index1 = index
        np.random.shuffle(index1)
        index2 = index1
        np.random.shuffle(index2)
        index3 = index2
        np.random.shuffle(index3)
        index_tripple = np.concatenate((index1, index2, index3), axis=0)
        images = []
        score_maps = []
        geo_maps = []
        training_masks = []
        for i in index_tripple:
            try:
                j = randint(0, frame_info[i]-1)
                video_path = FLAGS.data_path + '/' + video_set[i]+'.mp4'
                cap = cv2.VideoCapture(video_path)
                cap.set(1, j)
                ret, im = cap.read()
                h, w, _ = im.shape
                ######### add on-line data processing, Sep 5th by Xiaolong
                text_polys, text_tags = polys_array_list2[i][j], tags_array_list2[i][j]
                text_polys, text_tags = check_and_validate_polys_old(text_polys, text_tags, (h, w))
                if text_polys.shape[0] == 0:
                    continue
                rd_scale = np.random.choice(random_scale)
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
                text_polys *= rd_scale
                if np.random.rand() < background_ratio:
                    # crop background
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                    if text_polys.shape[0] > 0:
                        # cannot find background
                        continue
                    # pad and resize image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size))
                    score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                    geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                    geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                else:
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                    if text_polys.shape[0] == 0:
                        continue
                    h, w, _ = im.shape

                    # pad the image to the training input size or the longer side of image
                    new_h, new_w, _ = im.shape
                    max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = im_padded
                    # resize the image to input size
                    new_h, new_w, _ = im.shape
                    resize_h = input_size
                    resize_w = input_size
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    resize_ratio_3_x = resize_w/float(new_w)
                    resize_ratio_3_y = resize_h/float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape
                    score_map, geo_map, training_mask = generate_rbox((new_h, new_w), text_polys, text_tags)
                if vis:
                    fig, axs = plt.subplots(3, 2, figsize=(20, 30))
                    axs[0, 0].imshow(im[:, :, ::-1])
                    axs[0, 0].set_xticks([])
                    axs[0, 0].set_yticks([])
                    for poly in text_polys:
                        poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
                        poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
                        axs[0, 0].add_artist(Patches.Polygon(
                            poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
                        axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
                    axs[0, 1].imshow(score_map[::, ::])
                    axs[0, 1].set_xticks([])
                    axs[0, 1].set_yticks([])
                    axs[1, 0].imshow(geo_map[::, ::, 0])
                    axs[1, 0].set_xticks([])
                    axs[1, 0].set_yticks([])
                    axs[1, 1].imshow(geo_map[::, ::, 1])
                    axs[1, 1].set_xticks([])
                    axs[1, 1].set_yticks([])
                    axs[2, 0].imshow(geo_map[::, ::, 2])
                    axs[2, 0].set_xticks([])
                    axs[2, 0].set_yticks([])
                    axs[2, 1].imshow(training_mask[::, ::])
                    axs[2, 1].set_xticks([])
                    axs[2, 1].set_yticks([])
                    plt.tight_layout()
                    plt.show()
                    plt.close()

                if im is not None:
                    images.append(im[:, :, ::-1].astype(np.float32))
                    score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                    geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                    training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    yield images, score_maps, geo_maps, training_masks
                    images = []
                    score_maps = []
                    geo_maps = []
                    training_masks = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            t1 = time.time()
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
            print('Cost {} seconds to load data'.format((time.time() - t1)))
    finally:
        if enqueuer is not None:
            enqueuer.stop()

def get_batch_new(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator_fast(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

def get_batch_flow(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator_center_seq(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


def get_batch_seq(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator_fast_sequence(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    # 1. Generate images & GT with single thread;
    # 2. Run data loader in parallel by adopting existing framework;
    # data_generator = icdar.get_batch(num_workers=FLAGS.num_readers,
    #                                  input_size=FLAGS.input_size,
    #                                  batch_size=FLAGS.batch_size_per_gpu * len(gpus))
    # data = next(data_generator)
    config = get_config(FLAGS)
    ddd = generator_fast_sequence(config)
    print("Finishing !!!")
