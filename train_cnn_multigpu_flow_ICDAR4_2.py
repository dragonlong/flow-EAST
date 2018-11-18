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
     Nov. 9th, add new flow wrapping based on TF model;
With two references here:
https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/
https://gist.github.com/marta-sd/ba47a9626ae2dbcc47094c196669fd59
https://github.com/tensorflow/tensorflow/issues/3270
https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255
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
from tensorflow.python import pywrap_tensorflow
import sys
from copy import deepcopy
import pandas as pd
import seaborn as sns

from pwc.core_warp import dense_image_warp
from model import model_flow_east
from pwc.dataset_base import _DEFAULT_DS_VAL_OPTIONS
from pwc.dataset_mpisintel import MPISintelDataset
from pwc.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_VAL_OPTIONS
from pwc.visualize import display_img_pairs_w_flows
from pwc.net_options import cfg_lg
from pytorchpwc.utils import flow_inverse_warp
from utils import icdar
now = datetime.now()

############################# 1. Glabol Variables #########################################
################### 1.1 Training Setup
ICDAR2013 = '/media/dragonx/DataLight/ICDAR2013/'
ARC = '/media/dragonx/DataStorage/ARC/'
if platform.uname()[1] != 'dragonx-H97N-WIFI':
    print("Now it knows it's in a remote cluster")
    ARC = '/work/cascades/lxiaol9/ARC/'
    ICDAR2013 = '/work/cascades/lxiaol9/ARC/EAST/data/ICDAR2013/'
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 20, '')
tf.app.flags.DEFINE_integer('num_readers', 20, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9997, '')
tf.app.flags.DEFINE_string('gpu_list', '0,1', '')
tf.app.flags.DEFINE_string('data_path', ICDAR2013+'/train', 'data of ICDAR')
tf.app.flags.DEFINE_string('checkpoint_path', '/work/cascades/lxiaol9/ARC/EAST/checkpoints/FLOW_east/' + now.strftime("%Y%m%d-%H%M%S") +'/', '')
tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, '')
tf.app.flags.DEFINE_string('pretrained_model_path', ARC + "EAST/checkpoints/east/20180921-173054/", '')
tf.app.flags.DEFINE_string("prev_checkpoint_path", "/work/cascades/lxiaol9/ARC/EAST/checkpoints/FLOW_east/20181104-203818/", 'path' )
tf.app.flags.DEFINE_boolean('from_source',False, 'whether load data from source')
FLAGS = tf.app.flags.FLAGS
############### 1.2 GPUs, for EAST model
cfg_flow = cfg_lg()
gpus = list(range(len(FLAGS.gpu_list.split(','))))
east_opts = {
'verbose': True,
'ckpt_path': FLAGS.pretrained_model_path + 'model.ckpt-56092',
'batch_size': 20,
'gpu_devices': ['/device:GPU:0', '/device:GPU:1'],
# controller device to put the model's variables on (usually, /cpu:0 or /gpu:0 -> try both!)
'controller': '/device:CPU:0',
'x_dtype': tf.float32,  # image pairs input type
'x_shape': [512, 512, 3],  # image pairs input shape [2, H, W, 3]
'y_score_shape': [128, 128, 1],  # u,v flows output type
'y_geometry_shape': [128, 128, 5],  # u,v flows output shape [H, W, 2]
'x_mask_shape': [128, 128, 1]
}
############### 1.3 Configuration for Flow model
nn_opts = deepcopy(_DEFAULT_PWCNET_VAL_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = cfg_flow.ckpt_path
nn_opts['batch_size'] = 20      # Setting this to 1 leads to more accurate evaluations of the processing time
nn_opts['use_tf_data'] = False                   # Don't use tf.data reader for this simple task
nn_opts['gpu_devices'] = cfg_flow.gpu_devices    #
nn_opts['controller'] = cfg_flow.controller      # Evaluate on CPU or GPU?
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2
# The size of the images in this dataset are not multiples of 64
nn_opts['adapt_info'] = (1, 436, 1024, 2)

def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)
    # Placeholder needed by the new model >>>>>>>>>>>>>>> Output new score maps, and geometry maps
    input_feat_maps = tf.placeholder(tf.float32, shape=[cfg_flow.batch_size * cfg_flow.num_steps, 128, 128, 32], name='input_images')
    input_flow_maps = tf.placeholder(tf.float32, shape=[cfg_flow.batch_size * cfg_flow.num_steps , 128, 128, 2], name='input_flow_maps')
    input_score_maps = tf.placeholder(tf.float32, shape=[cfg_flow.batch_size, 128, 128, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[cfg_flow.batch_size, 128, 128, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[cfg_flow.batch_size, 128, 128, 8], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[cfg_flow.batch_size, 128, 128, 1], name='input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=5000, decay_rate=0.8, staircase=True)
    # add summary on learning rate
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # split only among the features, flow maps, and the GT labels
    input_feature_split = tf.split(input_feat_maps, len(gpus))
    input_score_maps_split = tf.split(input_score_maps, len(gpus))
    input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))
    input_flow_maps_split = tf.split(input_flow_maps, len(gpus))

    tower_grads = []
    reuse_variables = None
    tvars = []
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_feature_split[i]
                ifms = input_flow_maps_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                # create separate graph in different device
                total_loss, model_loss = tower_loss(iis, ifms, isms, igms, itms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True
                tvar1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tiny_embed')
                tvar2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred_module')
                tvars = tvar1 + tvar2
                grads = opt.compute_gradients(total_loss, tvars)
                tower_grads.append(grads)
    # #>>>>>>>>>>>>>>>>>>>>>>>>>>>> collect gradients from different devices and get the averaged gradient, large batch size
    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()
    g = tf.get_default_graph()
    with g.as_default():
        sess1 = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = FLAGS.prev_checkpoint_path + 'model.ckpt-14301'
            saver.restore(sess1, ckpt)
        else:
            sess1.run(init)
            # print('continue training from previous EAST weights')
            # ckpt = FLAGS.pretrained_model_path + 'model.ckpt-56092'
            # print('Restore from {}'.format(ckpt))
            # saver_alter.restore(sess1, ckpt)
    # further initialize the weights of detection head
    # reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.pretrained_model_path + 'model.ckpt-56092')
    # tensor_names=['feature_fusion/Conv_7/weights', 'feature_fusion/Conv_7/biases', 'feature_fusion/Conv_8/weights', 'feature_fusion/Conv_8/biases',
    #               'feature_fusion/Conv_9/weights', 'feature_fusion/Conv_9/biases']
    # variable_names = ['pred_module/Conv/weights', 'pred_module/Conv/biases', 'pred_module/Conv_1/weights', 'pred_module/Conv_1/biases',
    #                   'pred_module/Conv_2/weights', 'pred_module/Conv_2/biases']
    # # initialize the PWC-flow graph and weights here
    # for t in range(len(variable_names)):
    #     wt = reader.get_tensor(tensor_names[t]) # numpy array
    # # get the variables, or related rensors
    #     v1 = [var for var in tf.trainable_variables() if var.op.name==variable_names[t]]
    # # tf.assign(v1[0], w1) # won't work because you will add ops to the graph
    #     v1[0].load(wt, sess1)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> EAST model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    east_net = model_flow_east.EAST(mode='test', options=east_opts)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> PWCnet model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    nn = ModelPWCNet(mode='test', options=nn_opts)
    nn.print_config()
    data_generator = icdar.get_batch_flow(num_workers=FLAGS.num_readers, config=cfg_flow, is_training=True)
    start = time.time()
    # Start training
    for step in range(FLAGS.max_steps):
        data = next(data_generator)
        east_feed = np.reshape(data[0], [-1, 512, 512, 3])
        # data for flow net
        center_frame = np.array(data[0])[:, 2, :, : , :][:, np.newaxis, :, :, :]
        flow_feed_1 = np.reshape(np.tile(center_frame,(1,cfg_flow.num_steps,1,1,1)), [-1, 512, 512, 3])
        flow_feed = np.concatenate((east_feed[:, np.newaxis, :, :, :], flow_feed_1[:, np.newaxis, :, :, :]), axis = 1)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> feature extraction with EAST >>>>>>>>>>>>>>>>>>>>>>>> #
        # sometimes we need to run several rounds
        rounds = int(east_feed.shape[0]/40)
        feature_stack = []
        flow_maps_stack = []
        for r in range(rounds):
            feature_stack.append(east_net.sess.run([east_net.y_hat_test_tnsr], feed_dict={east_net.x_tnsr:east_feed[r*40:(r+1)*40, :, :, :]})[0][0])
        feature_maps = np.concatenate(feature_stack, axis=0)
        # >>>>>>>>>>>>>>>>>>>>>>>>>>> flow estimation with PWCnet >>>>>>>>>>>>>>>>>>>>>>>>> #
        # x: [batch_size,2,H,W,3] uint8; x_adapt: [batch_size,2,H,W,3] float32
        x_adapt, x_adapt_info = nn.adapt_x(flow_feed)
        if x_adapt_info is not None:
            y_adapt_info = (x_adapt_info[0], x_adapt_info[2], x_adapt_info[3], 2)
        else:
            y_adapt_info = None
        # Run the adapted samples through the network
        for r in range(rounds):
            feed_dict = {nn.x_tnsr: x_adapt[r*40:(r+1)*40, :, :, :, :]}
            y_hat = nn.sess.run(nn.y_hat_test_tnsr, feed_dict=feed_dict)
            y_hats, _ = nn.postproc_y_hat_test(y_hat, y_adapt_info)# suppose to be [batch, height, width, 2]
            flow_maps_stack.append(y_hats[:, 1::4, 1::4, :]/4)
        flow_maps = np.concatenate(flow_maps_stack, axis=0)
        # display_img_pairs_w_flows(img_pairs, pred_labels)
        with g.as_default():
            ml, tl, _ = sess1.run([model_loss, total_loss, train_op], feed_dict={input_feat_maps: feature_maps,
                                                                                input_score_maps: data[1],
                                                                                input_geo_maps: data[2],
                                                                                input_training_masks: data[3],
                                                                                input_flow_maps: flow_maps})
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
                saver.save(sess1, FLAGS.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % FLAGS.save_summary_steps == 0:
                _, tl, summary_str = sess1.run([train_op, total_loss, summary_op], feed_dict={input_feat_maps: feature_maps,
                                                                                              input_score_maps: data[1],
                                                                                              input_geo_maps: data[2],
                                                                                              input_training_masks: data[3],
                                                                                              input_flow_maps: flow_maps})
                summary_writer.add_summary(summary_str, global_step=step)


def embedding_net(feature):
    """
    embed features to a new space: 1x1x512, 3x3x512, 1x1x2048
    """
    num_outputs = [128, 256, 512]
    fe_1 = slim.conv2d(feature, num_outputs[0], [1, 1])
    fe_2 = slim.conv2d(fe_1, num_outputs[1], [3, 3])
    fe_3 = slim.conv2d(fe_2, num_outputs[2], [1, 1])
    return fe_3


def detector_top(feature):
    """
    apply detection head on updated features
    """
    F_score = slim.conv2d(feature, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
    # 4 channel of axis aligned bbox and 1 channel rotation angle
    geo_map = slim.conv2d(feature, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
    angle_map = (slim.conv2d(feature, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
    F_geometry = tf.concat([geo_map, angle_map], axis=-1)

    return F_score, F_geometry

# scale_by_y2 = tf.make_template('scale_by_y', my_op, scalar_name='y')
def tower_loss(feature, flow_maps, score_maps, geo_maps, training_masks, reuse_variables=None):
    """
    Multi-GPU training strategy:
    First split images/data to gpu first, then construct the model and loss;
    Input: image batch of 8 time steps,
           labels,
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        # the core part of the model
        c = np.zeros((cfg_flow.batch_size_per_gpu*cfg_flow.num_steps, 1), dtype=np.int32)
        num = range(2, cfg_flow.batch_size_per_gpu*cfg_flow.num_steps, cfg_flow.num_steps)
        for i in range(cfg_flow.batch_size_per_gpu):
            for j in range(cfg_flow.num_steps):
                c[i*5 + j] = num[i]
        indices = tf.constant(c)
        feature_midframe = tf.manip.gather_nd(feature, indices)
        # create a replicate of al center frames
        K = 2 # range of prev/after frames
        L = 5 # len of video seq
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            tiny_embed = tf.make_template('tiny_embed', embedding_net)
            # [batch_size*5, H, W, C]
            feature_w = flow_inverse_warp(feature, flow_maps)
            feature_w_e = tiny_embed(feature_w)
            feature_e = tiny_embed(feature_midframe)
            # [batch_size*5, H, W]
            eposilon = 1e-7
            weighting_params = tf.div(tf.reduce_sum(tf.multiply(feature_w_e, feature_e), axis=-1),  tf.multiply(tf.norm(feature_w_e, axis=-1), tf.norm(feature_e, axis=-1))+eposilon)
            # softmax across 5 frames [batch_size, 5, H, W, 1]
            weighting_normlized_reshape = tf.expand_dims(tf.nn.softmax(tf.reshape(weighting_params, [-1, 5, 128, 128]), axis=1), axis=4)
            feature_w_reshape = tf.reshape(feature_w, [-1, 5, 128, 128, 32])
            # sum ==> [batch_size, H, W, C]
            feature_fused = tf.reduce_sum(tf.multiply(feature_w_reshape, weighting_normlized_reshape), axis=1)
            # detection
            with tf.variable_scope('pred_module', reuse=reuse_variables):
                f_score, f_geometry = detector_top(feature_fused)
            # Loss
            model_loss = model_flow_east.loss(score_maps, f_score, geo_maps, f_geometry,training_masks)
            total_loss = tf.add_n([model_loss])
    #total_loss = tf.add_n([model_loss])
    # add summary
    if reuse_variables is None:
        tf.summary.image('raw_features', feature[:, :, :, 0:1])
        tf.summary.image('score_map', score_maps)
        tf.summary.image('score_map_pred', f_score * 255)
        tf.summary.image('geo_map_0', geo_maps[:, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', f_geometry[:, :, :, 0:1])
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        flag = True
        for g, _ in grad_and_vars:
            if g is not None:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            else:
                flag = False
        #
        if flag:
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

    return average_grads




if __name__ == '__main__':
    tf.app.run()
