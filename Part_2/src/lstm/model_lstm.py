# Demo on using LSTM, tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import lstm.utils.util as util

############ Macros ############
BASIC = "baisc"
CUDNN = "cudnn"
BLOCK = "block"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION  = "unidirection"

###################################################
# Model definition for video processing
###################################################


class VectModel(object):
    """Model used for PTB processing"""
    # here input is totally an object with all kinds of features created by Input class,
    # which use reader functions
    def __init__(self, is_training, config, input_data, targets):
        self._is_training = is_training
        self._rnn_params = None
        self._cell = None
        self._input_data = input_data
        self._targets = targets
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        size = config.hidden_size
        # vocab_size = config.vocab_size
        output_size = config.output_size
        # inputs dropout
        if is_training and config.keep_prob < 1:
            input_data = tf.nn.dropout(input_data, config.keep_prob)
            targets    = tf.nn.dropout(targets, config.keep_prob)

        # build up the model itself with lower-level function
        output, state = self._build_rnn_graph(input_data, config, is_training)

        softmax_w = tf.get_variable(
                "softmax_w", [size, output_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [output_size], dtype=tf.float32)
        pred      = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        # Reshape logits to be a 3-D tensor for sequence loss
        pred      = tf.reshape(pred, [self.batch_size, self.num_steps, output_size])
        ####
        loss = tf.square(pred - targets)
        # here used for tensorboard summary
        self._loss = loss
        self._cost = tf.reduce_mean(loss)
        self._final_state = state
        self._prediction = pred
        if not is_training:
            return

        # training details
        # since _lr is a variable, so we could assign number to it later by assignment
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        # gradient clipping
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars),config.max_grad_norm)
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        # how to manipulate the training gradient, the optimizer actually gives us an function to do that
        self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=tf.train.get_or_create_global_step())
        # so model also includes these options to get access to our training parameters, which mainly comes from
        # config function
        # self._grads = grads
        self._new_lr = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def _build_rnn_graph(self, inputs, config, is_training):
        if config.rnn_mode == CUDNN:
            return self._build_rnn_graph_cudnn(inputs, config, is_training)
        else:
            return self._build_rnn_graph_lstm(inputs, config, is_training)

    def _build_rnn_graph_cudnn(self, inputs, config, is_training):
        """Build the inference graph using CUDNN cell"""
        # here we want to pemute the dimensions
        inputs = tf.transpose(inputs, [1, 0, 2])
        self._cell = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=config.num_layers,
            num_units =config.hidden_size,
            input_size=config.vocab_size,
            dropout =1 - config.keep_prob if is_training else 0
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
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

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

    def import_ops(self, config):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            """ opaque_params,
                num_layers,
                num_units,
                input_size,
                input_mode=CUDNN_INPUT_LINEAR_MODE,
                direction=CUDNN_RNN_UNIDIRECTION,
                scope=None,
                name='cudnn_rnn_saveable'"""
            import pdb;pdb.set_trace()
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.CudnnLSTMSaveable(
                        opaque_params = None,
                        num_layers=config.num_layers,
                        num_units=config.hidden_size,
                        input_size=config.hidden_size,
                        input_mode=CUDNN_INPUT_LINEAR_MODE,
                        direction=CUDNN_RNN_UNIDIRECTION,
                        scope="Model/RNN",
                        name='cudnn_rnn_saveable'
                        )
                tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(util.with_prefix(self._name, "cost"))[0]
        num_replicas = FLAGS.num_gpus if self._name == "Train" else 1
        self._initial_state = util.import_state_tuples(
                self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = util.import_state_tuples(
                self._final_state, self._final_state_name, num_replicas)

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

    @property
    def prediction(self):
        return self._prediction
