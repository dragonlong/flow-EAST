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
     Nov. 13th, model changed to end-to-end by combing EAST with flow-head into one model
                        (some problems happen, so batch size is really small)
     Nov. 18th, modulize EAST again, and test tf.sigmoid activation;
     Nov. 22nd, recurrent layers with Conv-GRU;
     Nov. 24th, start training with Conv-GRU
With references here:
https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/
https://gist.github.com/marta-sd/ba47a9626ae2dbcc47094c196669fd59
https://github.com/tensorflow/tensorflow/issues/3270
"""
from __future__ import absolute_import, division, print_function
import time
import random
import numpy as np
import os
import cv2
from datetime import datetime
import platform
import logging
import sys
import GPUtil
from copy import deepcopy
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.contrib import slim
from model import model
from model import model_flow_east
from lstm import model_gru_agg
from pwc.model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_VAL_OPTIONS
from config.net_options import sys_cfg
from config.configrnn import get_config
from pytorchpwc.utils import flow_inverse_warp
from utils import icdar_light
now = datetime.now()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
############################# 1. Glabol Variables #########################################
################### 1.1 Training Setup
############ Macros ############
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
CONV  = "conv2d"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION = "unidirection"
ICDAR2013 = '/media/dragonx/DataLight/ICDAR2013/'
ARC = '/media/dragonx/DataStorage/ARC/'
if platform.uname()[1] != 'dragonx-H97N-WIFI':
    print("Now it knows it's in a remote cluster")
    ARC = '/work/cascades/lxiaol9/ARC/'
    ICDAR2013 = '/work/cascades/lxiaol9/ARC/EAST/data/ICDAR2013/'
#
tf.app.flags.DEFINE_string('mode', 'run', 'whether debugging or real running')
tf.app.flags.DEFINE_string('data_path', ICDAR2013+'/train', 'data of ICDAR')
# take care the parameters that are related to data inputs,
tf.app.flags.DEFINE_boolean('from_source', False, 'whether load data from source')
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 16, '')
tf.app.flags.DEFINE_integer('num_gpus', 2, '')
tf.app.flags.DEFINE_integer('num_steps', 5, '')
tf.app.flags.DEFINE_string('gpu_list', '0,1', '')
tf.app.flags.DEFINE_integer('num_readers', 28, '')

tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9997, '')

tf.app.flags.DEFINE_string('pretrained_model_path', ARC + "EAST/checkpoints/east/20180921-173054/model.ckpt-56092", 'EAST ckpt')
tf.app.flags.DEFINE_string("prev_checkpoint_path", "/work/cascades/lxiaol9/ARC/EAST/checkpoints/GRU_east/20181025-235849/-14001", 'AGG ckpt' )
tf.app.flags.DEFINE_string('checkpoint_path', '/work/cascades/lxiaol9/ARC/EAST/checkpoints/GRU_agg/' + now.strftime("%Y%m%d-%H%M%S") +'/', '')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_string('flownet_type', "small", '')
tf.app.flags.DEFINE_string("model", "test", "A type of model. Possible options are: small, medium, large.")
tf.app.flags.DEFINE_string("rnn_mode", CONV, "one of CUDNN: BASIC, BLOCK")
tf.app.flags.DEFINE_boolean("random", True, "style when feeding grouped frames data")
tf.app.flags.DEFINE_string('geometry', 'RBOX', 'which geometry to generate, RBOX or QUAD')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 400, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 10, '')
tf.app.flags.DEFINE_string('apply_grad', 'top', '')
FLAGS = tf.app.flags.FLAGS

#
def main(argv=None):
    m_cfg = sys_cfg()
    config = get_config(FLAGS)
    config.batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus
    config.num_layers = 3
    config.num_steps  = 5
    #
    eval_config = get_config(FLAGS)
    eval_config.batch_size = 2
    eval_config.num_layers = 3
    eval_config.num_steps  = 5
#============================ I. Model options ==============================#
#>>>>>>>>>>>>>>>for PWCnet module network
    nn_opts = deepcopy(_DEFAULT_PWCNET_VAL_OPTIONS)
    if FLAGS.flownet_type is 'small':
        nn_opts['use_dense_cx'] = False
        nn_opts['use_res_cx']   = False
        nn_opts['pyr_lvls']     = 6
        nn_opts['flow_pred_lvl']= 2
        nn_opts['ckpt_path']    = '/work/cascades/lxiaol9/ARC/PWC/checkpoints/pwcnet-sm-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-592000' # Model to eval
    else:
        nn_opts['use_dense_cx'] = True
        nn_opts['use_res_cx']   = True
        nn_opts['pyr_lvls']     = 6
        nn_opts['flow_pred_lvl']= 2
        nn_opts['ckpt_path']    = '/work/cascades/lxiaol9/ARC/PWC/checkpoints/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

    nn_opts['verbose']     = True
    nn_opts['batch_size']  = 32     # This is Batch_size per GPU(16*4/2/2 = 16)
    nn_opts['use_tf_data'] = False  # Don't use tf.data reader for this simple task
    nn_opts['gpu_devices'] = ['/device:GPU:0', '/device:GPU:1']    #
    nn_opts['controller']  = '/device:CPU:0'  # Evaluate on CPU or GPU?
    nn_opts['adapt_info']  = (1, 436, 1024, 2)
    nn_opts['x_shape']     = [2, 512, 512, 3] # image pairs input shape [2, H, W, 3]
    nn_opts['y_shape']     = [512, 512, 2] # u,v flows output shape [H, W, 2]
    #>>>>>>>>>>>>>>>> For EAST module network
    east_opts = {
    'verbose': True,
    'ckpt_path': FLAGS.pretrained_model_path,
    'batch_size': 40,
    'batch_size_per_gpu': 20,
    'gpu_devices': ['/device:GPU:0', '/device:GPU:1'],
    # controller device to put the model's variables on (usually, /cpu:0 or /gpu:0 -> try both!)
    'controller': '/device:CPU:0',
    'x_dtype': tf.float32,     # image pairs input type
    'x_shape': [512, 512, 3],  # image pairs input shape [2, H, W, 3]
    'y_score_shape': [128, 128, 1],     # u,v flows output type
    'y_geometry_shape': [128, 128, 5],  # u,v flows output shape [H, W, 2]
    'x_mask_shape': [128, 128, 1]
    }
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    else:
        if not FLAGS.restore:
            tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
            tf.gfile.MkDir(FLAGS.checkpoint_path)
#=============================== II. building graph for east + agg =================================#
    # 1.1 Input placeholders
    batch_size = FLAGS.batch_size_per_gpu * FLAGS.num_gpus
    len_seq = FLAGS.num_steps
    # input_images = tf.placeholder(tf.float32, shape=[batch_size*len_seq, 512, 512, 3], name='input_images')
    input_feat_maps = tf.placeholder(tf.float32, shape=[batch_size, len_seq, 128, 128, 32], name='input_feature_maps')
    input_flow_maps = tf.placeholder(tf.float32, shape=[batch_size, len_seq-1, 128, 128, 2], name='input_flow_maps')
    input_score_maps = tf.placeholder(tf.float32, shape=[batch_size , len_seq, 128, 128, 1], name='input_score_maps')
    if FLAGS.geometry == 'RBOX':
        input_geo_maps = tf.placeholder(tf.float32, shape=[batch_size, len_seq, 128, 128, 5], name='input_geo_maps')
    else:
        input_geo_maps = tf.placeholder(tf.float32, shape=[batch_size, len_seq, 128, 128, 8], name='input_geo_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[batch_size, len_seq, 128, 128, 1], name='input_training_masks')
    # 1.2 lr & opt
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.8, staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate)
    # 1.3 add summary
    tf.summary.scalar('learning_rate', learning_rate)
    # tf.summary.image('input_images', input_images[2:20:5, :, :, :])
    # 1.4 build graph in tf
    # input_images_split     = tf.split(input_images, FLAGS.num_gpus)
    input_feature_split = tf.split(input_feat_maps,  FLAGS.num_gpus)
    input_score_maps_split = tf.split(input_score_maps, FLAGS.num_gpus)
    input_geo_maps_split   = tf.split(input_geo_maps, FLAGS.num_gpus)
    input_training_masks_split = tf.split(input_training_masks, FLAGS.num_gpus)
    input_flow_maps_split  = tf.split(input_flow_maps, FLAGS.num_gpus)
    tower_grads = []
    reuse_variables = None
    tvars = []
    gpus = list(range(len(FLAGS.gpu_list.split(','))))
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_feature_split[i]
                ifms = input_flow_maps_split[i]
                isms = input_score_maps_split[i]
                igms = input_geo_maps_split[i]
                itms = input_training_masks_split[i]
                # model changed to recurrent one, we only need the recurrent loss returned
                total_loss, model_loss = model_gru_agg.tower_loss(iis, ifms, isms, igms, itms, gpu_id=gpu_id, config=config, reuse_variables=reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True
                # tvar1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tiny_embed')
                # tvar2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred_module')
                # tvars = tvar1 + tvar2
                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)
    # 1.5 gradient parsering
    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # 1.6 get training operations
    summary_op = tf.summary.merge_all()
    # variable_averages = tf.train.ExponentialMovingAverage(
    #     FLAGS.moving_average_decay, global_step)
    # variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')
    # 1.8 Saver & Session & Restore
    saver = tf.train.Saver(tf.global_variables())
    # sv = tf.train.Supervisor()
    summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())
    init = tf.global_variables_initializer()
    g = tf.get_default_graph()
    with g.as_default():
        config1 = tf.ConfigProto()
        config1.gpu_options.allow_growth = True
        config1.allow_soft_placement = True
        sess1 = tf.Session(config=config1)
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = FLAGS.prev_checkpoint_path
            saver.restore(sess1, ckpt)
        else:
            sess1.run(init)
            # var_list1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='multi_rnn_cell')
            # var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred_module')
            # var_list_part1 = var_list1 + var_list2
            # saver_alter1 = tf.train.Saver({v.op.name: v for v in var_list_part1})
            # # var_list3 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tiny_embed')
            # # var_list4 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred_module')
            # # var_list_part2 = var_list3 + var_list4
            # # saver_alter2 = tf.train.Saver({v.op.name: v for v in var_list_part2})
            # print('continue training from previous weights')
            # ckpt1 = FLAGS.prev_checkpoint_path
            # print('Restore from {}'.format(ckpt1))
            # saver_alter1.restore(sess1, ckpt1)
            # # print('continue training from previous Flow weights')
            # # ckpt2 = FLAGS.prev_checkpoint_path
            # # print('Restore from {}'.format(ckpt2))
            # # saver_alter2.restore(sess1, ckpt2)
#============================= III. Other necessary componets before training =============================#
    print("Step 1: AGG model has been reconstructed")
    GPUtil.showUtilization()
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> EAST model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
    east_net = model_flow_east.EAST(mode='test', options=east_opts)
    print("Step 2: EAST model has been reconstructed")
    GPUtil.showUtilization()
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> PWCnet model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    nn = ModelPWCNet(mode='test', options=nn_opts)
    print("Step 3: PWC model has been reconstructed")
    GPUtil.showUtilization()
    train_data_generator = icdar_light.get_batch_seq(num_workers=FLAGS.num_readers, config=config, is_training=True)
    # val_data_generator = icdar.get_batch_seq(num_workers=FLAGS.num_readers, config=eval_config, is_training=False)
    start = time.time()
#============================= IV. Training over Steps(!!!)================================================#
    print("Now we're starting training!!!")
    for step in range(FLAGS.max_steps):
    #>>>>>>>>>>>>> data
        if FLAGS.mode == "debug":
            data = []
            data.append(np.ones((config.batch_size, FLAGS.num_steps, 512, 512, 3), dtype=np.float32))
            data.append(np.ones((batch_size , len_seq, 128, 128, 1), dtype=np.float32))
            data.append(np.ones((batch_size , len_seq, 128, 128, 5), dtype=np.float32))
            data.append(np.ones((batch_size , len_seq, 128, 128, 1), dtype=np.float32))
        else:
            data = next(train_data_generator)
        if step < 3:
            print("Data ready!!!")
        east_feed = np.reshape(data[0], [-1, 512, 512, 3])
        target_frame = np.reshape(np.array(data[0])[:, 0:4, :, : , :], [-1, 512, 512, 3])
        source_frame = np.reshape(np.array(data[0])[:, 1:5, :, : , :], [-1, 512, 512, 3])
        flow_feed = np.concatenate((source_frame[:, np.newaxis, :, :, :], target_frame[:, np.newaxis, :, :, :]), axis = 1)
        flow_maps_stack = []
    # >>>>>>>>>>>>>>>>>>>>>>>>>>> feature extraction with EAST >>>>>>>>>>>>>>>>>>>>>>>> #
        rounds = int(east_feed.shape[0]/east_opts['batch_size'])
        feature_stack = []
        flow_maps_stack = []
        for r in range(rounds):
            feature_stack.append(east_net.sess.run([east_net.y_hat_test_tnsr], feed_dict={east_net.x_tnsr:east_feed[r*east_opts['batch_size']:(r+1)*east_opts['batch_size'], :, :, :]})[0][0])
        feature_maps = np.concatenate(feature_stack, axis=0)
        feature_maps_reshape = np.reshape(feature_maps, [-1, config.num_steps, 128, 128, 32])
    #>>>>>>>>>>>>>>> flow estimation with PWCnet
        # x: [batch_size,2,H,W,3] uint8; x_adapt: [batch_size,2,H,W,3] float32
        x_adapt, x_adapt_info = nn.adapt_x(flow_feed)
        if x_adapt_info is not None:
            y_adapt_info = (x_adapt_info[0], x_adapt_info[2], x_adapt_info[3], 2)
        else:
            y_adapt_info = None
        mini_batch = nn_opts['batch_size']*nn.num_gpus
        rounds = int(flow_feed.shape[0]/mini_batch)
        for r in range(rounds):
            feed_dict = {nn.x_tnsr: x_adapt[r*mini_batch:(r+1)*mini_batch, :, :, :, :]}
            y_hat = nn.sess.run(nn.y_hat_test_tnsr, feed_dict=feed_dict)
            if FLAGS.mode == "debug":
                print("Step 5: now finish running one round of PWCnet for flow estimation")
                GPUtil.showUtilization()
            y_hats, _ = nn.postproc_y_hat_test(y_hat, y_adapt_info)# suppose to be [batch, height, width, 2]
            flow_maps_stack.append(y_hats[:, 1::4, 1::4, :]/4)
        flow_maps = np.concatenate(flow_maps_stack, axis=0)
        print("flow maps has shape ", flow_maps.shape[:])
        flow_maps = np.reshape(flow_maps, [-1, FLAGS.num_steps-1, 128, 128, 2])
    #>>>>>>>>>>>>>>> running training session
        with g.as_default():
            ml, tl, _ = sess1.run([model_loss, total_loss, train_op], \
                                        feed_dict={input_feat_maps: feature_maps_reshape,
                                                   input_score_maps: data[1],
                                                   input_geo_maps: data[2],
                                                   input_training_masks: data[3],
                                                   input_flow_maps: flow_maps
                                                   })
            if FLAGS.mode == "debug":
                print("Step 6: running one round on training!!!")
                GPUtil.showUtilization()
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
                _, tl, summary_str = sess1.run([train_op, total_loss, summary_op], feed_dict={input_feat_maps: feature_maps_reshape,
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
    fe_1 = slim.conv2d(feature, num_outputs[0], [1, 1], activation_fn=tf.nn.sigmoid, normalizer_fn=None)
    fe_2 = slim.conv2d(fe_1, num_outputs[1], [3, 3], activation_fn=tf.nn.sigmoid, normalizer_fn=None)
    fe_3 = slim.conv2d(fe_2, num_outputs[2], [1, 1], activation_fn=tf.nn.sigmoid, normalizer_fn=None)
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


def tower_loss(feature, flow_maps, score_maps, geo_maps, training_masks, cfg_flow, reuse_variables=None):
    """
    Multi-GPU training strategy:
    First split images/data to gpu first, then construct the model and loss;
    Input: image batch of 8 time steps,
           labels,
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
    #>>>>>>>>>>>>>>>>> the core >>>>>>>>>>>>>>>>>#
        # feature = model_flow_east.model(images, is_testing=True)
        c = np.zeros((FLAGS.batch_size_per_gpu*cfg_flow.num_steps, 1), dtype=np.int32)
        num = range(2, FLAGS.batch_size_per_gpu*cfg_flow.num_steps, cfg_flow.num_steps)
        for i in range(FLAGS.batch_size_per_gpu):
            for j in range(cfg_flow.num_steps):
                c[i*cfg_flow.num_steps + j] = num[i]
        indices = tf.constant(c)
        feature_midframe = tf.manip.gather_nd(feature, indices)
        # create a replicate of al center frames
        L = FLAGS.num_steps # len of video seq
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
            weighting_normlized_reshape = tf.expand_dims(tf.nn.softmax(tf.reshape(weighting_params, [-1, L, 128, 128]), axis=1), axis=4)
            feature_w_reshape = tf.reshape(feature_w, [-1, L, 128, 128, 32])
            # sum ==> [batch_size, H, W, C]
            feature_fused = tf.reduce_sum(tf.multiply(feature_w_reshape, weighting_normlized_reshape), axis=1)
            # detection
            with tf.variable_scope('pred_module', reuse=reuse_variables):
                f_score, f_geometry = detector_top(feature_fused)
            # Loss
            model_loss = model_flow_east.loss(score_maps, f_score, geo_maps, f_geometry,training_masks)
            total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
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
