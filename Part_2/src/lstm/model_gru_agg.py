# Demo on using LSTM, tensorflow
"""
LOG:
Sep. 13th: add graph freeze on the feature fusion + ConvLSTM part
Sep. 23th: add tensorboard visualization on sub loss
Dec. 7th : no score map loss;

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.slim as slim
import inspect
import tensorflow as tf
import _init_paths
import utils.util as util
from lstm.cell import ConvGRUCell
from model import modelEAST
from config.configrnn import get_config
from pytorchpwc.utils import flow_inverse_warp


import numpy as np
############ Macros ############
BASIC = "baisc"
CUDNN = "cudnn"
BLOCK = "block"
CONV  = "conv2d"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION = "bidirection"
CUDNN_RNN_UNIDIRECTION  = "unidirection"

FLAGS = tf.app.flags.FLAGS
###################################################
# Model definition for video processing
###################################################
# get rnn cell
def get_gru_cell(config, is_training):
    if config.rnn_mode == BASIC:
        return tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                           reuse = not is_training)
    if config.rnn_mode == BLOCK:
        return tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias = 0.0)
    if config.rnn_mode == CONV:
        # kwargs = {"conv_ndims": 3, "input_shape": config.shape, "output_channels": config.filters,
        #           "kernel_shape": config.kernel}
        # return tf.contrib.rnn.Conv2DLSTMCell(**kwargs)
        return ConvGRUCell([128, 128], config.filters, config.kernel, activation=tf.nn.relu)
    raise ValueError("rnn_mode %s not supported" % config.rnn_mode)


def build_model(inputs, flow_maps, config=None, reuse_variables=None):
    score_map_set = []
    geo_map_set = []
    state   = None
    # initialize with global setup
    reuse_variables = reuse_variables
    for k in range(config.num_steps):
        F_score, F_geometry, state = gru_agg(inputs[:, k, :, :, :], state, config, reuse_variables, lstm=True)
        score_map_set.append(F_score[0, :, :, :])
        geo_map_set.append(F_geometry[0, :, :, :])
        if k < config.num_steps-1:
            state = (flow_inverse_warp(state[0], flow_maps[:, k, :, :, :]), flow_inverse_warp(state[1], \
                            flow_maps[:, k, :, :, :]), flow_inverse_warp(state[2], flow_maps[:, k, :, :, :]))
        reuse_variables=True
    return score_map_set, geo_map_set

def tower_loss(inputs, flow_maps, score_maps, geo_maps, training_masks, gpu_id=None, config=None, reuse_variables=None):
    """
    input with input features, flow estimaation for score maps, geo maps
    """
    loss_set   = []
    L_Dice_set = []
    L_AABB_set = []
    L_Theta_set = []
    score_map_set = []
    geo_map_set = []
    state   = None
    # initialize with global setup
    reuse_variables = reuse_variables
    for k in range(config.num_steps):
        F_score, F_geometry, state = gru_agg(inputs[:, k, :, :, :], state, config, reuse_variables, lstm=True)
        l_cls, l_aabb, l_theta, l_model = loss(score_maps[:, k, :, :, :], F_score,
                                               geo_maps[:, k, :, :, :], F_geometry,
                                               training_masks[:, k, :, :, :])
        L_Dice_set.append(l_cls)
        L_AABB_set.append(l_aabb)
        L_Theta_set.append(l_theta)
        loss_set.append(l_model)
        score_map_set.append(F_score[0, :, :, 0:1])
        geo_map_set.append(F_geometry[0, :, :, 0:1])
        # warp for every early step until last
        if k < config.num_steps-1:
            state = (flow_inverse_warp(state[0], flow_maps[:, k, :, :, :]), flow_inverse_warp(state[1], \
                            flow_maps[:, k, :, :, :]), flow_inverse_warp(state[2], flow_maps[:, k, :, :, :]))
        reuse_variables=True
    loss_cls   = tf.reduce_mean(tf.stack(L_Dice_set, 0))
    loss_aabb  = tf.reduce_mean(tf.stack(L_AABB_set, 0))
    loss_theta = tf.reduce_mean(tf.stack(L_Theta_set, 0))
    loss_model = tf.reduce_mean(tf.stack(loss_set, 0))
    cost = loss_model
    # self._loss = tf.reshape(self._loss, [self.batch_size, self.num_steps, `])
    # try to transform the loss array with tf.reshape
    final_state = state
    if gpu_id == 0:
        tf.summary.image('score_map', score_maps[0, :, :, :, 0]* 255)
        tf.summary.image('score_map_pred', tf.stack(score_map_set, 0) * 255)
        tf.summary.image('geo_map_0', geo_maps[0, :, :, :, 0:1])
        tf.summary.image('geo_map_0_pred', tf.stack(geo_map_set, 0))
        tf.summary.image('training_mask', training_masks[0, :, :, :, 0]*255)
        tf.summary.scalar('score_loss', loss_cls)
        tf.summary.scalar('geometry_loss', loss_aabb)
        tf.summary.scalar('angle_loss', loss_theta)
        tf.summary.scalar('model_loss', loss_model)
        tf.summary.scalar('total_loss', cost)
    return cost, loss_model


def gru_agg(input, state, config, reuse_variables, is_training=True, lstm=True):
    def make_cell():
        cell = get_gru_cell(config, is_training)
        if is_training and config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
        return cell
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        if lstm:
            cell = tf.contrib.rnn.MultiRNNCell(
                [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
            initial_state = cell.zero_state(FLAGS.batch_size_per_gpu, tf.float32)
            if state is None:
                state = initial_state
            output, state = cell(input, state)
            # F_heatmap = slim.conv2d(output, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
            with tf.variable_scope('pred_module', reuse=reuse_variables):
                F_score = slim.conv2d(output, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                # output1 = slim.conv2d(output, 32, 1, activation_fn=tf.sigmoid, normalizer_fn=None)
                geo_map = slim.conv2d(output, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * 512
                # add a separate layer for further regression
                angle_map = (slim.conv2d(output, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)
            return F_score, F_geometry, state
        else:
            pass


def loss(y_true_cls, y_pred_cls,
         y_true_geo, y_pred_geo,
         training_mask):
    '''
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    '''
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
    h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -tf.log((area_intersect + 1.0)/(area_union + 1.0))
    L_theta = 1 - tf.cos(theta_pred - theta_gt)
    L_g = L_AABB + 20 * L_theta

    return classification_loss, tf.reduce_mean(L_AABB * y_true_cls * training_mask), tf.reduce_mean(L_theta * y_true_cls * training_mask), tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss



def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)

    return loss


if __name__ == "__main__":
    flags = tf.app.flags
    # The only part you need to modify during training
    flags.DEFINE_string("system", "local", "deciding running env")
    flags.DEFINE_string("data_path", "/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process", "Where data is stored")
    flags.DEFINE_string('checkpoint_path', '/media/dragonx/DataStorage/ARC/checkpoints/', '')
    flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
    flags.DEFINE_string("rnn_mode", CONV, "one of CUDNN: BASIC, BLOCK")
    flags.DEFINE_integer("num_readers", 4, "process used to fetch data")
    flags.DEFINE_string("model", "test", "A type of model. Possible options are: small, medium, large.")
    flags.DEFINE_integer("num_gpus", 1, "Larger than 1 will create multiple training replicas")
    flags.DEFINE_boolean("random", True, "style when feeding grouped frames data")
    flags.DEFINE_boolean("source", False, "whether load data from source")
    flags.DEFINE_boolean("dis_plt", False, "whether using pyplot real-time display ")
    flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
    flags.DEFINE_integer('save_summary_steps', 100, '')
    flags.DEFINE_string('pretrained_model_path', '/media/dragonx/DataStorage/ARC/EASTRNN/weights/EAST/resnet_v1_50.ckpt', '')
    flags.DEFINE_string('geometry', 'RBOX', 'set for bb')
    FLAGS = flags.FLAGS

    save_path = '/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/LSTM/'
    #train_input = DetectorInputMul(save_path, 1, 2, 0)
    print("data has been loaded")
    config = get_config(FLAGS)
    # Global initializer for Variables in the model
    gpu_options = tf.GPUOptions(allow_growth=True)
    #global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    # log: May 3rd, we need to adapt the model input, with config
    with tf.name_scope("Train"):
        # use placeholder to stand for input and targets
        initializer = tf.random_normal_initializer()
        x_train = tf.placeholder(tf.float32, shape=[None, config.num_steps, None, None, 3])
        m = ArrayModel(True, config, x_train, reuse_variables=None, initializer=initializer)
    with tf.name_scope("Val"):
        # use placeholder to stand for input and targets
        initializer = tf.random_normal_initializer()
        x_val = tf.placeholder(tf.float32, shape=[None, config.num_steps, None, None, 3])
        mval = ArrayModel(True, config, x_val, reuse_variables=True, initializer=initializer)

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path, slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            initializer = tf.random_normal_initializer()
            sess.run(initializer)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

    print('hahaha')
