##===================== Statements & Copyright ===================##
"""
AUTHOR:  Xiaolong Li, VT
CONTENT: Used for Video-text project
LOG: Sep. 13th Change the data loader, freeze part of graph,
     Sep. 18th Check variable names, initialize Variables in Scope "feature_fusion", and "resnet_v1_50"
     Sep. 23th Add Val branch, regularization loss, and tensorboard visualization,
     Oct. 4th  Re-train from ICDAR2013 checkpoints
     Oct. 5th  Fine-tuning with ICDAR2015
     Oct. 7th  This script using for ICDAR 2013 only
     Oct. 31th Add pwc optical flow for feature aggregation
     Nov. 1st All operations are manupolating on graph
     Nov. 3rd, compare to ICDAR1, add detect_head initialization, and gradient only apply to tiny_embed, pred_module;
     Nov. 3rd, compare to ICDAR2, the batch_size is increased from 8 to 16;
     Nov. 4th, Compared to ICDAR3, EAST is treated as a separate module;
     Nov. 5th, compared to ICDAR4, it restarts, with 0.00001 lr
     Nov. 6th, change into totally OOP style
     Nov. 10th, test code on evaluation,
     Nov. 13th, change the EAST to original one due to better performance
With two references here:
https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/
https://gist.github.com/marta-sd/ba47a9626ae2dbcc47094c196669fd59
https://github.com/tensorflow/tensorflow/issues/3270
"""
from __future__ import absolute_import, division, print_function
import time
import random
import numpy as np
import os
from datetime import datetime
import platform
import logging
import tensorflow as tf
from tensorflow.contrib import slim
import sys
from copy import deepcopy
import pandas as pd
import seaborn as sns

from model import model
from model import model_flow_east
from pwc.dataset_base import _DEFAULT_DS_VAL_OPTIONS
from pwc.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_VAL_OPTIONS
from pwc.net_options import cfg_lg, cfg_sm
from model.model_aggregation import Aggregation
from utils import icdar
now = datetime.now()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
############################# 1. Glabol Variables #########################################
################### 1.1 Training Setup
ICDAR2013 = '/media/dragonx/DataLight/ICDAR2013/'
ARC = '/media/dragonx/DataStorage/ARC/'
if platform.uname()[1] != 'dragonx-H97N-WIFI':
    print("Now it knows it's in a remote cluster")
    ARC = '/work/cascades/lxiaol9/ARC/'
    ICDAR2013 = '/work/cascades/lxiaol9/ARC/EAST/data/ICDAR2013/'
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 8, '')
tf.app.flags.DEFINE_integer('seq_len', 5, '')
tf.app.flags.DEFINE_integer('num_readers', 20, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9997, '')
tf.app.flags.DEFINE_string('gpu_list', '0,1', '')
tf.app.flags.DEFINE_integer('num_gpus', 2, '')
tf.app.flags.DEFINE_string('data_path', ICDAR2013+'/train', 'data of ICDAR')
tf.app.flags.DEFINE_string('checkpoint_path', '/work/cascades/lxiaol9/ARC/EAST/checkpoints/FLOW_east/' + now.strftime("%Y%m%d-%H%M%S") +'/', '')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, '')
tf.app.flags.DEFINE_string('apply_grad', 'top', '')
# Model 1: EAST
tf.app.flags.DEFINE_string('pretrained_model_path', ARC + "EAST/checkpoints/east/20180921-173054/model.ckpt-56092", '')
# Model 2: PWC-net
tf.app.flags.DEFINE_string('flownet_type', "large", '')
# Model 3: AGG
tf.app.flags.DEFINE_string("prev_checkpoint_path", "/work/cascades/lxiaol9/ARC/EAST/checkpoints/FLOW_east/20181104-203818/model.ckpt-14301", 'path' )
tf.app.flags.DEFINE_boolean('from_source',False, 'whether load data from source')
FLAGS = tf.app.flags.FLAGS

#
def main(argv=None):
    ######################## Set up model configurations ######################
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Agg model options >>>>>>>>>>>>>>>>>>>>>>>>>>
    aggre_opts = {
    'verbose': True,
    'ckpt_path': FLAGS.prev_checkpoint_path,
    'batch_size_per_gpu': FLAGS.batch_size_per_gpu,
    'gpu_devices': ['/device:GPU:0', '/device:GPU:1'],
    # controller device to put the model's variables on (usually, /cpu:0 or /gpu:0 -> try both!)
    'controller': '/device:GPU:0',
    'train_mode': 'fine-tune'
    }
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>> EAST model options >>>>>>>>>>>>>>>>>>>>>>>>>>
    gpus = list(range(len(FLAGS.gpu_list.split(','))))
    checkpoint_path = '/work/cascades/lxiaol9/ARC/EAST/checkpoints/east/'
    idname1 = '20180921-173054'
    idname2 = 'model.ckpt-56092'
    model_path = checkpoint_path + idname1 + '/' + idname2
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PWCnet model options >>>>>>>>>>>>>>>>>>>>>>>>>>
    nn_opts = deepcopy(_DEFAULT_PWCNET_VAL_OPTIONS)
    if FLAGS.flownet_type is 'small':
        pwc_cfg = cfg_sm()
        pwc_cfg.num_steps = FLAGS.seq_len
        nn_opts['use_dense_cx'] = False
        nn_opts['use_res_cx'] = False
        nn_opts['pyr_lvls'] = 6
        nn_opts['flow_pred_lvl'] = 2
    else:
        pwc_cfg = cfg_lg()
        pwc_cfg.num_steps = FLAGS.seq_len
        nn_opts['use_dense_cx'] = True
        nn_opts['use_res_cx'] = True
        nn_opts['pyr_lvls'] = 6
        nn_opts['flow_pred_lvl'] = 2
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = pwc_cfg.ckpt_path
    nn_opts['batch_size'] = 20      # This is Batch_size per GPU
    nn_opts['use_tf_data'] = False  # Don't use tf.data reader for this simple task
    nn_opts['gpu_devices'] = pwc_cfg.gpu_devices    #
    nn_opts['controller'] = pwc_cfg.controller      # Evaluate on CPU or GPU?
    nn_opts['adapt_info'] = (1, 436, 1024, 2)
    nn_opts['x_shape'] = [2, 512, 512, 3] # image pairs input shape [2, H, W, 3]
    nn_opts['y_shape'] = [512, 512, 2] # u,v flows output shape [H, W, 2]
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)
    # Create new model for the aggregation model
    agg = Aggregation(mode='train_noval', options=aggre_opts)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> PWCnet model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    nn = ModelPWCNet(mode='test', options=nn_opts)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> EAST model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    gpu_options = tf.GPUOptions(allow_growth=True)
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    _, _, feature_raw = model.model(input_images, is_training=False)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    #>>>>>>>>>>>>>>>>>>>>>>>>  3. restore the model from weights  >>>>>>>>>>>>>#
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)
    data_generator = icdar.get_batch_flow(num_workers=FLAGS.num_readers, config=pwc_cfg, is_training=True)
    start = time.time()
    # Start training
    for step in range(FLAGS.max_steps):
        data = next(data_generator)
        east_feed = np.reshape(data[0], [-1, 512, 512, 3])
        # data for flow net
        center_frame = np.array(data[0])[:, 2, :, : , :][:, np.newaxis, :, :, :]
        flow_feed_1 = np.reshape(np.tile(center_frame,(1,pwc_cfg.num_steps,1,1,1)), [-1, 512, 512, 3])
        flow_feed = np.concatenate((flow_feed_1[:, np.newaxis, :, :, :], east_feed[:, np.newaxis, :, :, :]), axis = 1)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> feature extraction with EAST >>>>>>>>>>>>>>>>>>>>>>>> #
        # sometimes we need to run several rounds
        mini_batch = 40
        rounds = int(east_feed.shape[0]/mini_batch)
        feature_stack = []
        flow_maps_stack = []
        for r in range(rounds):
            feature = sess.run([feature_raw],feed_dict={input_images: east_feed[r*mini_batch:(r+1)*mini_batch, :, :, :]})
            feature_stack.append(feature[0])
        feature_maps = np.concatenate(feature_stack, axis=0)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> flow estimation with PWCnet >>>>>>>>>>>>>>>>>>>>>>>>> #
        # x: [batch_size,2,H,W,3] uint8; x_adapt: [batch_size,2,H,W,3] float32
        x_adapt, x_adapt_info = nn.adapt_x(flow_feed)
        if x_adapt_info is not None:
            y_adapt_info = (x_adapt_info[0], x_adapt_info[2], x_adapt_info[3], 2)
        else:
            y_adapt_info = None
        # Run the adapted samples through the network
        mini_batch = nn_opts['batch_size']*nn.num_gpus
        rounds = int(flow_feed.shape[0]/mini_batch)
        for r in range(rounds):
            feed_dict = {nn.x_tnsr: x_adapt[r*mini_batch:(r+1)*mini_batch, :, :, :, :]}
            y_hat = nn.sess.run(nn.y_hat_test_tnsr, feed_dict=feed_dict)
            y_hats, _ = nn.postproc_y_hat_test(y_hat, y_adapt_info)# suppose to be [batch, height, width, 2]
            flow_maps_stack.append(y_hats[:, 1::4, 1::4, :]/4)
        flow_maps = np.concatenate(flow_maps_stack, axis=0)
        # display_img_pairs_w_flows(img_pairs, pred_labels)
        with agg.graph.as_default():
            ml, tl, _ = agg.sess.run([agg.model_loss, agg.total_loss, agg.train_op], feed_dict={agg.input_feat_maps: feature_maps,
                                                                                            agg.input_score_maps: data[1],
                                                                                            agg.input_geo_maps: data[2],
                                                                                            agg.input_training_masks: data[3],
                                                                                            agg.input_flow_maps: flow_maps})
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break
            if step % 10 == 0:
                avg_time_per_step = (time.time() - start)/10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus))/(time.time() - start)
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                    step, ml, tl, avg_time_per_step, avg_examples_per_second))

            if step % FLAGS.save_checkpoint_steps == 0:
                agg.saver.save(agg.sess, FLAGS.checkpoint_path + 'model.ckpt', global_step=agg.global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, _, summary_str = agg.sess.run([agg.model_loss, agg.total_loss, agg.summary_op], feed_dict={agg.input_feat_maps: feature_maps,
                                                                                                    agg.input_score_maps: data[1],
                                                                                                    agg.input_geo_maps: data[2],
                                                                                                    agg.input_training_masks: data[3],
                                                                                                    agg.input_flow_maps: flow_maps})
                agg.summary_writer.add_summary(summary_str, global_step=step)

if __name__ == '__main__':
    tf.app.run()
