##===================== Statements & Copyright ===================##
"""
AUTHOR:  Xiaolong Li, VT
CONTENT: Used for Video-text project
LOG: Sep. 13th Change the data loader, freeze part of graph,
     Sep. 18th Check variable names, initialize Variables in Scope "feature_fusion", and "resnet_v1_50"
     Sep. 23th Add Val branch, regularization loss, and tensorboard visualization,
"""
# Demo on using LSTM, tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import collections
import io
import platform
from datetime import datetime
from random import randint
import logging
from utils.rnn_eval import model_eval
from utils.rnn_eval import draw_illu
import utils.util as util
from utils import icdar
from config.configrnn import get_config
from lstm.model_rnn_east import ArrayModel

############ Macros ############
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
CONV  = "conv2d"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION = "unidirection"

###############################
# FLAGS or args.parser
#####################################################
from tensorflow.python.client import device_lib
flags = tf.flags
now = datetime.now()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# The only part you need to modify during training
flags.DEFINE_float('learning_rate', 0.00001, '')
flags.DEFINE_string("data_path", "/media/dragonx/DataStorage/ARC/EASTRNN/data/ICDAR/train", "Where data is stored" )
flags.DEFINE_string("save_path", "/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/LSTM_east/"+ now.strftime("%Y%m%d-%H%M%S") + '/', "Model output")
flags.DEFINE_string("base_path", "/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process", "root directory for data")
flags.DEFINE_string("vis_path", "/media/dragonx/DataStorage/ARC/EASTRNN/vis/LSTM/"+now.strftime("%Y%m%d-%H%M%S"), "save visualization")
flags.DEFINE_string("video_path", "/media/dragonx/DataStorage/Video Detection/ICDAR/train/", "where video files are stored")
flags.DEFINE_string('pretrained_model_path', '/work/cascades/lxiaol9/ARC/EAST/checkpoints/east/20180921-135717/', '')
flags.DEFINE_string("checkpoints_path", "/home/dragonx/Documents/VideoText2018/EAST-master/weights/east_icdar2015_resnet_v1_50_rbox/", "checkpoints")
# Model running infos
flags.DEFINE_string("system", "local", "deciding running env")
flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
flags.DEFINE_boolean('partially_restore', True, 'whether to restore the weights of back-bone')
flags.DEFINE_string("model", "test", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_integer("num_gpus", 1, "Larger than 1 will create multiple training replicas")
flags.DEFINE_string("rnn_mode", CONV, "one of CUDNN: BASIC, BLOCK")
flags.DEFINE_boolean("random", True, "style when feeding grouped frames data")
flags.DEFINE_boolean('from_source', False, 'whether load data from source')
flags.DEFINE_integer('video_end', 10, 'number of videos we use for training')
# visualization
flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
flags.DEFINE_integer('perform_val_steps', 20, '')
flags.DEFINE_boolean("dis_plt", False, "whether using pyplot real-time display ")
flags.DEFINE_boolean("img_save", True, "whether we need to save the visualization")
flags.DEFINE_integer('dis_freq', 1, "frequency on training epoch when visualization")
flags.DEFINE_integer("dis_step", 20, "at No. step to display")
flags.DEFINE_integer("num_readers", 24, "process used to fetch data")

FLAGS = flags.FLAGS
##############################


def run_step(session, model, data,  config, step,  eval_op=None, verbose=False, summary_writer=None):
    """
    :param session:
    :param model :  train/val/test
    :param input_:  class with .data, heat_maps, .cnt_frame, .video_name
    :param config:  configuration for all parameters
    :param epoch:
    :param eval_op: whether or not to train and back-propagate
    :param verbose:
    :param summary_writer: checkpoints and data for tensorboard visualization
    :return:       loss over epoch
    """
    costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    # tensors dict to run
    fetches = {
            "cost": model.cost,
            "final_state": model.final_state,
            "loss": model.loss,
            "summary": model.summary_merged,
            }
    if eval_op is not None:
        fetches["eval_op"] = eval_op

    feed_dict = {}
    feed_dict[model.input_data] = data[0]
    feed_dict[model.input_score_maps] = data[1]
    feed_dict[model.input_geo_maps] = data[2]
    feed_dict[model.input_training_masks] = data[3]
    for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h
    vals = session.run(fetches, feed_dict=feed_dict)
    cost = vals["cost"]
    state = vals["final_state"]
    frame_set =[]
    summary = vals["summary"]
    summary_writer.add_summary(summary, step + 1)
    return (cost)


def main(_):
    # check the running platform
    if platform.uname()[1] != 'dragonx-H97N-WIFI':
        print("Now it knows it's in a remote cluster")
        FLAGS.system = "remote"
        FLAGS.data_path = "/home/lxiaol9/ARC/EASTRNN/data/ICDAR/train"
        FLAGS.vis_path = "/home/lxiaol9/ARC/EASTRNN/vis/LSTM/"
        FLAGS.save_path = "/work/cascades/lxiaol9/ARC/EAST/checkpoints/LSTM_east/" + now.strftime("%Y%m%d-%H%M%S")
        FLAGS.video_path = "/home/lxiaol9/ARC/EASTRNN/data/ICDAR2013/train/"
        FLAGS.checkpoints_path = FLAGS.save_path
        FLAGS.pretrained_model_path = "/work/cascades/lxiaol9/ARC/EAST/checkpoints/east/20180921-173054/"
    print("############## Step1: The environment path has been set up ###############")
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if (FLAGS.num_gpus > len(gpus)):
        raise ValueError("Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus    config = get_config(FLAGS)
    config.batch_size = 8
    config.num_layers = 3
    config.num_steps  = 10))
    # Model parameters using config
    config = get_config(FLAGS)
    config.batch_size = 8
    config.num_layers = 3
    config.num_steps  = 10
    eval_config = get_config(FLAGS)
    eval_config.batch_size = 8
    eval_config.num_layers = 3
    eval_config.num_steps  = 10
    with tf.Graph().as_default():
        # Global initializer for Variables in the model
        # log: May 3rd, we need to adapt the model input, with config
        with tf.name_scope("Train"):
            # use placeholder to stand for input and targets
            initializer = tf.random_normal_initializer()
            x_train = tf.placeholder(tf.float32, shape=[None, config.num_steps, None, None, 3])
            m = ArrayModel(True, config, x_train, reuse_variables=None, initializer=initializer)
            print("finished Training model generation")
            training_score = tf.summary.image('score_map', m.input_score_maps[0, :, :, :, :])
            training_score_pred = tf.summary.image('score_map_pred', tf.stack(m.score_map_set, 0) * 255)
            training_cost_sum = tf.summary.scalar("Loss", m.cost)
            training_lr = tf.summary.scalar("Learning_Rate", m.lr)
            training_input = tf.summary.image('input_images', m.input_data[0, :, :, :, :] )
            training_loss_aabb = tf.summary.scalar('geometry_AABB', m.loss_aabb)
            training_loss_theta = tf.summary.scalar('geometry_theta',m.loss_theta)
            training_loss_cls = tf.summary.scalar('classification_dice_loss', m.loss_cls)
            m.summary_merged = tf.summary.merge([training_lr, training_cost_sum, training_loss_aabb, training_loss_theta, training_loss_cls, training_input, training_score, training_score_pred])
        with tf.name_scope("Val"):
            # use placeholder to stand for input and targets
            initializer = tf.random_normal_initializer()
            x_val = tf.placeholder(tf.float32, shape=[None, eval_config.num_steps, None, None, 3])
            mvalid = ArrayModel(True, eval_config, x_val, reuse_variables=True, initializer=initializer)
            val_cost_sum = tf.summary.scalar("Loss", mvalid.cost)
            val_score = tf.summary.image('score_map', mvalid.input_score_maps[0, :, :, :, :])
            val_score_pred = tf.summary.image('score_map_pred', tf.stack(mvalid.score_map_set, 0) * 255)
            val_input = tf.summary.image('input_images', mvalid.input_data[0, :, :, :, :] )
            val_loss_aabb = tf.summary.scalar('geometry_AABB', mvalid.loss_aabb)
            val_loss_theta = tf.summary.scalar('geometry_theta',mvalid.loss_theta)
            val_loss_cls = tf.summary.scalar('classification_dice_loss', mvalid.loss_cls)
            mvalid.summary_merged = tf.summary.merge([val_cost_sum, val_score, val_loss_aabb, val_loss_theta, val_loss_cls, val_score_pred, val_input])

        # Now we have got our models ready, so create a dictionFLAGSary to store those computational graph
        models = {"Train": m}
        # Module 2
        print("#############Step 2: models has been built############")
        for name, model in models.items():
            model.export_ops(name)
        metagraph = tf.train.export_meta_graph()
        soft_placement = False
        # we could also do coding in parallel
        if FLAGS.num_gpus > 1:
            soft_placement = True
            util.auto_parallel(metagraph, m)
        # if FLAGS.pretrained_model_path is not None:
        #     checkpoint_path = FLAGS.pretrained_model_path
        #     ckpt_state = tf.train.get_checkpoint_state(checkpoint_path )
        #     ckpt = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        #     # 1. one way is to wrap original weights and change the variable names
        #     variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])
        #     variable_restore_op = slim.assign_from_checkpoint(ckpt, slim.get_trainable_variables(),
        #                                                          ignore_missing_vars=True)
        # Sep-18th, Try using a different saver to only initialize part of the model
        var_list1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feature_fusion')
        var_list2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_v1_50')
        var_list = var_list1 + var_list2
        saver_alter = tf.train.Saver({v.op.name: v for v in var_list})
        config_proto = tf.ConfigProto(allow_soft_placement = soft_placement)
        #########################changes made up here######################
        train_data_generator = icdar.get_batch_seq(num_workers=FLAGS.num_readers, config=config, is_training=True)
        val_data_generator = icdar.get_batch_seq(num_workers=FLAGS.num_readers, config=eval_config, is_training=False)
        print("##############Step 3: Heatmap, GT data is ready now################")
        sv = tf.train.Supervisor()
        with sv.managed_session(config=config_proto) as session:
            if FLAGS.restore:
                print('continue training from previous checkpoint')
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoints_path)
                sv.saver.restore(session, ckpt)
            elif FLAGS.partially_restore:
                print('continue training from previous EAST checkpoint')
                ckpt = FLAGS.pretrained_model_path + 'model.ckpt-56092'
                logger.info('Restore from {}'.format(ckpt))
                saver_alter.restore(session, ckpt)
            else:
                if FLAGS.pretrained_model_path is not None:
                    variable_restore_op(session)
            train_writer = tf.summary.FileWriter(FLAGS.save_path + '/train/', session.graph)
            val_writer = tf.summary.FileWriter(FLAGS.save_path + '/val/')
            print("###########Step 4 : start training. ###########")
            for i in range(config.max_steps):
                lr_decay = config.lr_decay**max(i + 1 - config.max_steps, 0.0)
                m.assign_lr(session, FLAGS.learning_rate * lr_decay)
                data_train = next(train_data_generator)
                # apply training along the way
                print("Step: %d Learning Rate: %.5f" % (i+1, session.run(m.lr)))
                train_loss = run_step(session, m, data_train, config, i, eval_op = m.train_op, summary_writer= train_writer, verbose = True)
                print("Step: %d training loss: %.5f" % (i+1, train_loss))
                if i % FLAGS.perform_val_steps == 0:
                    data_val = next(val_data_generator)
                    valid_loss = run_step(session, mvalid, data_val, eval_config, i,  summary_writer= val_writer, verbose = True)
                    print("Step: %d Valid loss: %.5f" % (i+1, valid_loss))
                if (i % FLAGS.save_checkpoint_steps == 0) and FLAGS.save_path:
                    print("Saving model to %s." % FLAGS.save_path)
                    sv.saver.save(session, FLAGS.save_path, global_step = sv.global_step)


if __name__ == '__main__':
    tf.app.run()
