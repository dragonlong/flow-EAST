"""
=====================================================
# This code is written using skicit-video module, runs both EAST
# text detector in scene image, and a RNN-CNN text detector in python
# by
=====================================================

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
import xmltodict
import tensorflow as tf
import modelGAP
from icdar import restore_rectangle
import lanms
from evalgap import resize_image, sort_poly, detect
from lstm.posteval import eval_single_frame
# from lstm.utils.reader import read_XML
# import lstm.utils.reader as reader
# import lstm.utils.util as util
# from root folder
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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


def eval_score():

    return


def main():
    checkpoint_path = '/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/EAST/GAP/checkpoints_arc/'
    global_path = '/media/dragonx/752d26ef-8f47-416d-b311-66c6dfabf4a3/Video_text/ICDAR/train/'
    save_path = '/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process/'
    items = os.listdir('/media/dragonx/DataStorage/ARC/EASTRNN/data/icdar2015/video/')
    newlist = []
    for names in items:
        if names.endswith('.avi'):
            newlist.append(os.path.splitext(names)[0])

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', default=checkpoint_path)
    args = parser.parse_args()

    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(checkpoint_path))
    # read images until it is complete
    logger.info('loading model')
    gpu_options = tf.GPUOptions(allow_growth=True)
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry, v_feature = modelGAP.model(input_images, is_training=False)
    #
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    # restore the model from weights
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)
    sum = 0
    # get infos for video written
    for sample in newlist:
        print('##############Processing '+sample+'###############')
        filename = global_path + sample + '.mp4'
        XML_filepath = global_path + sample + '_GT.xml'
        print(XML_filepath)
        cap = cv2.VideoCapture(filename)
        frame_width =  int(cap.get(3))
        frame_height = int(cap.get(4))
        # read ground-truth boxes
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter(save_path+sample+'.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        index = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            index = index+1
            if ret == True:
                # targets are actually dict relates to one frame
                target = read_xml_solo(XML_filepath, index)
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
                score_m, geometry, feature = sess.run(
                    [f_score, f_geometry, v_feature],
                    feed_dict={input_images: [im_resized[:,:,::-1]]})
                timer['net'] = time.time() - start
                print('score shape {:s}, geometry shape {:s}, feature shape {:s}'.format(str(score_m.shape), str(geometry.shape), str(feature.shape)))
                boxes, timer = detect(score_map=score_m, geo_map=geometry, timer=timer)
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
                    # 'rtparams': rtparams,
                    # 'timing': timer,
                    # 'geometry': geometry,
                    # 'score':float(score),
                }
                # 1. print boxes number
                print('%d Boxs found'%(len(text_lines)))
                # 2. eval_single_frame(target, box)
                p, r, f1 = eval_single_frame(target, ret)
                print('Precision %f, recall %f, F_measure %f' % (p, r, f1))
                # 3. save files into directory
                jsonfile = json.dumps(ret)
                directory = save_path+sample
                if not os.path.exists(directory):
                    os.makedirs(directory+'/json/')
                    os.makedirs(directory + '/npy/')
                    os.makedirs(directory + '/score/')

                jsonfname = directory+'/json/frame'+format(index, '03d')+'.json'
                npyname   = directory+'/npy/frame'+format(index, '03d')+'.npy'
                scorename = directory + '/score/frame' + format(index, '03d') + '.npy'
                np.save(npyname, feature)
                np.save(scorename, score_m)
                f = open(jsonfname,"w")
                f.write(jsonfile)
                f.close()
                # visualization
                new_img = draw_illu(img.copy(), ret)
                new_img1 = draw_illu_gt(new_img.copy(), target, p, r, f1)
                cv2.imshow('Annotated Frame with EAST', new_img1)
                out.write(new_img1)
                # Quit when Q is pressed
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                time.sleep(0.02)
            else:
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
