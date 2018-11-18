"""
=======================================================================================
# This code runs with model.py, and process Video_16_3_2.mp4 with GT+Pred bounding box;
# Originally adapted in March, modified in Sep.
- Draw boxes
- Model deplyment
- Video Generation
=======================================================================================
"""
import cv2
import os
import time
import datetime
import numpy as np
import uuid
import json
import functools
import logging
import collections
import argparse
import tensorflow as tf

import _init_paths
from model import model
from utils.icdar import restore_rectangle

import lanms
from utils.eval import resize_image, sort_poly, detect

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def main():
    #checkpoint_path = '/home/dragonx/Documents/VideoText2018/EAST-master/weights/east_icdar2015_resnet_v1_50_rbox/'
    filename        = '/media/dragonx/752d26ef-8f47-416d-b311-66c6dfabf4a3/Video Detection/ICDAR/train/Video_16_3_2.mp4'

    cap             = cv2.VideoCapture(filename)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', default=checkpoint_path)
    args = parser.parse_args()

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(checkpoint_path))
    # read images until it is completed
    index = 0
    logger.info('loading model')

    gpu_options = tf.GPUOptions(allow_growth=True)
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    # restore the model from weights
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)
    # get infos for video written
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('EAST_testDemo1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        index = index+1
        if ret == True:
            cv2.imshow('Frame', frame)
            print('Processing %d frame with '%(index), frame.shape)
            ######### Use EAST text detector ###########
            start_time = time.time()
            img = frame
            rtparams = collections.OrderedDict()
            rtparams['start_time'] = datetime.datetime.now().isoformat()
            rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
            timer = collections.OrderedDict([
                ('net', 0),
                ('restore', 0),
                ('nms', 0)
            ])

            im_resized, (ratio_h, ratio_w) = resize_image(img)
            rtparams['working_size'] = '{}x{}'.format(
                im_resized.shape[1], im_resized.shape[0])
            start = time.time()
            score, geometry = sess.run(
                [f_score, f_geometry],
                feed_dict={input_images: [im_resized[:,:,::-1]]})
            timer['net'] = time.time() - start

            boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
            logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

            if boxes is not None:
                scores = boxes[:,8].reshape(-1)
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

            duration = time.time() - start_time
            timer['overall'] = duration
            logger.info('[timing] {}'.format(duration))

            text_lines = []
            if boxes is not None:
                text_lines = []
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
                'rtparams': rtparams,
                'timing': timer,
            }


            new_img = draw_illu(img.copy(), ret)
            cv2.imshow('Annotated Frame with EAST', new_img)
            out.write(new_img)
            # Quit when Q is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            time.sleep(.100)
        else:
            break


    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
