# Demo on using LSTM, tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.contrib.slim as slim
import inspect
import tensorflow as tf
import _init_paths
import utils.util as util
from lstm.cell import ConvLSTMCell
from utils.input_node import DetectorInputMul
from model import modelEAST
from config.configrnn import get_config

import numpy as np
############ Macros ############
BASIC = "baisc"
CUDNN = "cudnn"
BLOCK = "block"
CONV  = "conv2d"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION = "bidirection"
CUDNN_RNN_UNIDIRECTION  = "unidirection"
###################################################
# Model definition for video processing
###################################################
def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

#from pdb import set_trace as pb
def pb():
    print('running through debugging line ', lineno())


class ArrayModel(object):
    """Model used for PTB processing"""
    # here input is totally an object with all kinds of features created by Input class,
    # which use reader functions
    def __init__(self, is_training, config, input_data, reuse_variables=None, initializer=None):
        self.input_score_maps = tf.placeholder(tf.float32, shape=[None, None, 128, 128, 1], name='input_score_maps')
        if config.geometry == 'RBOX':
            self.input_geo_maps = tf.placeholder(tf.float32, shape=[None, None,  128, 128, 5], name='input_geo_maps')
        else:
            self.input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, 128, 128, 8], name='input_geo_maps')
        self.input_training_masks = tf.placeholder(tf.float32, shape=[None, None, 128, 128, 1], name='input_training_masks')
        self._is_training = is_training
        self._rnn_params = None
        self._cell = None
        self._input_data = input_data
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        # Note here the input is [batch_size, time_steps, [shape]+channels]
        if is_training and config.keep_prob < 1:
            input_data = tf.nn.dropout(input_data, config.keep_prob)
        # build up the model itself with lower-level function
        loss_set = []
        state = None
        reuse_variables = reuse_variables
        for k in range(config.num_steps):
            F_score, F_geometry, state = self._cnn_lstm(input_data[:, k, :, :, :], state, config, reuse_variables, lstm=True)
            # loss_set.append(loss(tf.squeeze(self.input_score_maps[:, k, :, :, :], axis=1), F_score[lbo:hbo, :, :, :],
            #                      tf.squeeze(self.input_geo_maps[:, k, :, :, :], axis=1), F_geometry[lbo:hbo, :, :, :],
            #                      tf.squeeze(self.input_training_masks[:, k, :, :, :], axis=1)))
            loss_set.append(loss(self.input_score_maps[:, k, :, :, :], F_score,
                                 self.input_geo_maps[:, k, :, :, :], F_geometry,
                                 self.input_training_masks[:, k, :, :, :]))
            reuse_variables=True
                # add summary
            if reuse_variables is None:
                tf.summary.image('score_map', self.input_score_maps[:, k, :, :, :])
                tf.summary.image('score_map_pred', F_score * 255)
                tf.summary.image('geo_map_0', self.input_geo_maps[:, k, :, :, 0:1])
                tf.summary.image('geo_map_0_pred', F_geometry[:, :, :, 0:1])
                tf.summary.image('training_masks', self.input_training_masks[:, k, :, :, :])
        self._loss = tf.stack(loss_set, 0)
        self._cost = tf.reduce_mean(tf.stack(loss_set, 0))
        # self._loss = tf.reshape(self._loss, [self.batch_size, self.num_steps, `])
        # try to transform the loss array with tf.reshape
        self._final_state = state
        if not is_training:
            return

        # training details
        # since _lr is a variable, so we could assign number to it later by assignment
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # gradient clipping
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars),config.max_grad_norm)
        # optimizer
        # optimizer = tf.train.GradientDescentOptimizer(self._lr)
        optimizer = tf.train.AdamOptimizer(self._lr)
        # how to manipulate the training gradient, the optimizer actually gives us an function to do that
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())
        self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    # for cnn+lstm model design
    def _cnn_lstm(self, inputs, state, config, reuse_variables, is_training = True, lstm=True):
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            f_score, f_geometry, feature = modelEAST.model(inputs, is_training=False)
            if lstm:
                cell = tf.contrib.rnn.MultiRNNCell(
                    [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)
                self._initial_state = cell.zero_state(config.batch_size, tf.float32)
                if state is None:
                    state = self._initial_state
                output, state = cell(feature, state)
                F_score = slim.conv2d(output, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle, what is the scale for text
                geo_map = slim.conv2d(output, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * 512
                angle_map = (slim.conv2d(output, 1, 1, activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None) - 0.5) * np.pi / 2  # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)

                return F_score, F_geometry, state
            else:
                pass

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell"""
        # here we want to pemute the dimensions
        # inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers= config.num_layers,
            num_units = config.hidden_size,
            input_size= config.vocab_size,
            dropout = 1 - config.keep_prob if is_training else 0
        )
        # what is this used for
        #params_size_t = self.
        params_size_t = self._cell.params_size()
        self._rnn_params = tf.get_variable(
                "lstm_params",
                initializer=tf.random_uniform(
                        [params_size_t], -config.init_scale, config.init_scale),
                validate_shape=False)
        c = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                                  tf.float32)
        h = tf.zeros([config.num_layers, self.batch_size, config.hidden_size],
                                  tf.float32)
        self._initial_state = (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)
        outputs, h, c = self._cell(inputs, h, c, self._rnn_params, is_training)
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, config.hidden_size])
        return outputs, (tf.contrib.rnn.LSTMStateTuple(h=h, c=c),)

    def _get_lstm_cell(self, config, is_training):
        if config.rnn_mode == BASIC:
            return tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True,
                                               reuse = not is_training)
        if config.rnn_mode == BLOCK:
            return tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias = 0.0)
        if config.rnn_mode == CONV:
            # kwargs = {"conv_ndims": 3, "input_shape": config.shape, "output_channels": config.filters,
            #           "kernel_shape": config.kernel}
            # return tf.contrib.rnn.Conv2DLSTMCell(**kwargs)
            return ConvLSTMCell([128, 128], config.filters, config.kernel)
        raise ValueError("rnn_mode %s not supported" % config.rnn_mode)

    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonial LSTM cells."""
        """Self defined functions """
        def make_cell():
            cell = self._get_lstm_cell(config, is_training)
            # when a cell is constructed, we will need to use the mechanism called wrapper
            if is_training and config.keep_prob < 1:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)
            return cell
        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True
        )
        self._initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self._initial_state
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :, :, :], state)
                outputs.append(cell_output)
        # pb()
        output = tf.concat(outputs, 0)
        # Performs fully dynamic unrolling of inputs
        # output, state = tf.nn.dynamic_rnn(cell=cell,
        #                            inputs=data,
        #                            dtype=tf.float32)

        return output, state

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        # import pdb;pdb.set_trace()
        ops = {util.with_prefix(self._name, "cost"): self._cost}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr, lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = util.with_prefix(self._name, "initial")
        self._final_state_name = util.with_prefix(self._name, "final")
        util.export_state_tuples(self._initial_state, self._initial_state_name)
        util.export_state_tuples(self._final_state, self._final_state_name)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def loss(self):
        return self._loss

    # @property
    # def grads(self):
    #     return self._grads
    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name


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
    tf.summary.scalar('classification_dice_loss', loss)
    return loss


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
    tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
    tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
    L_g = L_AABB + 20 * L_theta

    return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss


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
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

    print('hahaha')
