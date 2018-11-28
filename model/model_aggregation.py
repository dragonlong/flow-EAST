import tensorflow as tf
import numpy as np
import platform
from tensorflow.contrib import slim
from tensorflow.python import pywrap_tensorflow
#
import _init_paths
from nets import resnet_v1
from pwc.core_warp import dense_image_warp
from pytorchpwc.utils import flow_inverse_warp


FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bwilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


class Aggregation(object):
    """
    A self-contained module for all use
    """
    def __init__(self, name='train', mode=None, options=None, session=None):
        # Class Attributes
        assert(mode in ['train_noval', 'train_with_val', 'val', 'val_notrain', 'test'])
        self.mode, self.opts, self.sess= mode, options, session
        self.y_hat_train_tensor = self.y_hat_val_tensor = self.y_hat_test_tensor = None
        self.name = name
        self.gpus = list(range(len(FLAGS.gpu_list.split(','))))
        self.num_gpus = len(self.gpus)
        # Build the graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        # Create the session
        with self.graph.as_default():
            self.config_session(session)
            # build graph
            self.build_graph()

    def config_session(self, sess):
        """Configure a TF session, if one doesn't already exist.
        Args:
            sess: optional TF session
        """
        if sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            self.sess = tf.Session(config=config)
        else:
            self.sess = sess


    def build_graph(self):
        """ Build the complete graph in TensorFlow
        """
        # with tf.device(self.main_device):
        # Configure input and output tensors
        self.config_placeholders()
        # Build the backbone network, then:
        # In training mode, configure training ops (loss, metrics, optimizer, and lr schedule)
        # Also, config train logger and, optionally, val logger
        # In validation mode, configure validation ops (loss, metrics)
        if self.mode in ['train_noval', 'train_with_val']:
            if self.num_gpus == 1:
                self.build_model()
                self.config_train_ops()
            else:
                self.build_model_towers_loss()

        elif self.mode in ['val', 'val_notrain']:
            if self.num_gpus == 1:
                self.build_model()
                self.setup_metrics_ops()
            else:
                self.build_model_towers()
        else:  # inference mode
            if self.num_gpus == 1:
                self.build_model()
            else:
                self.build_model_towers()

        # Set output tensors
        self.set_output_tnsrs()
        # Init saver (override if you wish) and load checkpoint if it exists
        self.init_saver()
        self.load_ckpt()


    def config_placeholders(self):
        """Configure input and output tensors
        Args:
            x_dtype, x_shape:  type and shape of elements in the input tensor
            y_dtype, y_shape:  shape of elements in the input tensor
        """
        # Increase the batch size with the number of GPUs dedicated to computing TF ops
        batch_size = self.num_gpus * FLAGS.batch_size_per_gpu * FLAGS.seq_len
        batch_size_val = self.num_gpus * FLAGS.batch_size_per_gpu
        self.input_feat_maps = tf.placeholder(tf.float32, shape=[batch_size, 128, 128, 32], name='input_feat_images')
        self.input_flow_maps = tf.placeholder(tf.float32, shape=[batch_size , 128, 128, 2], name='input_flow_maps')
        self.input_score_maps = tf.placeholder(tf.float32, shape=[batch_size_val, 128, 128, 1], name='input_score_maps')
        if FLAGS.geometry == 'RBOX':
            self.input_geo_maps = tf.placeholder(tf.float32, shape=[batch_size_val, 128, 128, 5], name='input_geo_maps')
        else:
            self.input_geo_maps = tf.placeholder(tf.float32, shape=[batch_size_val, 128, 128, 8], name='input_geo_maps')
        self.input_training_masks = tf.placeholder(tf.float32, shape=[batch_size_val, 128, 128, 1], name='input_training_masks')


    # Necessary methods for public usage
    def build_model(self):
        reuse_variables = None
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            # the core part of the model
            batch_size = FLAGS.batch_size_per_gpu*FLAGS.seq_len
            c = np.zeros((batch_size, 1), dtype=np.int32)
            num = range(2, batch_size, FLAGS.seq_len)
            for i in range(FLAGS.batch_size_per_gpu):
                for j in range(FLAGS.seq_len):
                    c[i*FLAGS.seq_len+ j] = num[i]
            indices = tf.constant(c)
            feature = self.input_feat_maps
            flow_maps = self.input_flow_maps
            feature_midframe = tf.manip.gather_nd(feature, indices)
            # create a replicate of al center frames
            K = 2 # range of prev/after frames
            L = 5 # len of video seq
            with tf.variable_scope(tf.get_variable_scope(), reuse=None):
                tiny_embed = tf.make_template('tiny_embed', embedding_net)
                # [batch_size*5, H, W, C]
                feature_w = flow_inverse_warp(feature, flow_maps)
                feature_w_e = tiny_embed(feature_w)
                feature_e = tiny_embed(feature_midframe)
                # [batch_size*5, H, W]
                eposilon = 1e-7
                weighting_params = tf.div(tf.reduce_sum(tf.multiply(feature_w_e, feature_e), axis=-1),  tf.multiply(tf.norm(feature_w_e, axis=-1), tf.norm(feature_e, axis=-1))+eposilon)
                # softmax across 5 frames [batch_size, 5, H, W, 1]
                weighting_normlized_reshape = tf.expand_dims(tf.nn.softmax(tf.reshape(weighting_params, [-1, FLAGS.seq_len, 128, 128]), axis=1), axis=4)
                feature_w_reshape = tf.reshape(feature_w, [-1, FLAGS.seq_len, 128, 128, 32])
                # sum ==> [batch_size, H, W, C]
                feature_fused = tf.reduce_sum(tf.multiply(feature_w_reshape, weighting_normlized_reshape), axis=1)
                # detection
                with tf.variable_scope('pred_module', reuse=None):
                    self.f_score, self.f_geometry = detector_top(feature_fused)
                self.feature_wrapped = feature_w
                self.feature_new = feature_fused
                # Loss

        #total_loss = tf.add_n([model_loss])
        # add summary
        if reuse_variables is None:
            tf.summary.image('raw_features', feature[:, :, :, 0:1])
            tf.summary.image('score_map_pred', self.f_score * 255)
            tf.summary.image('geo_map_0_pred', self.f_geometry[:, :, :, 0:1])


    def build_model_towers_loss(self):
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step, decay_steps=5000, decay_rate=0.8, staircase=True)
        # add summary on learning rate
        tf.summary.scalar('learning_rate', learning_rate)
        self.optim = tf.train.AdamOptimizer(learning_rate)
        # split only among the features, flow maps, and the GT labels
        input_feature_split = tf.split(self.input_feat_maps, self.num_gpus)
        input_score_maps_split = tf.split(self.input_score_maps, self.num_gpus)
        input_geo_maps_split = tf.split(self.input_geo_maps, self.num_gpus)
        input_training_masks_split = tf.split(self.input_training_masks, self.num_gpus)
        input_flow_maps_split = tf.split(self.input_flow_maps, self.num_gpus)

        tower_grads = []
        reuse_variables = None
        tvars = []
        for i, gpu_id in enumerate(self.gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('model_%d' % gpu_id) as scope:
                    iis = input_feature_split[i]
                    ifms = input_flow_maps_split[i]
                    isms = input_score_maps_split[i]
                    igms = input_geo_maps_split[i]
                    itms = input_training_masks_split[i]
                    # create separate graph in different device
                    self.total_loss, self.model_loss = tower_loss(iis, ifms, isms, igms, itms, reuse_variables)
                    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                    reuse_variables = True
                    grads = self.optim.compute_gradients(self.total_loss)
                    tower_grads.append(grads)
        # #>>>>>>>>>>>>>>>>>>>>>>>>>>>> collect gradients from different devices and get the averaged gradient, large batch size
        grads = average_gradients(tower_grads)
        apply_gradient_op = self.optim.apply_gradients(grads, global_step=self.global_step)
        self.summary_op = tf.summary.merge_all()
        # save moving average
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, self.global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        # batch norm updates
        with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
            self.train_op = tf.no_op(name='train_op')
        # self.summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    def build_model_towers(self):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=5000, decay_rate=0.5, staircase=True)
        # add summary on learning rate
        tf.summary.scalar('learning_rate', learning_rate)
        self.optim = tf.train.AdamOptimizer(learning_rate)
        # split only among the features, flow maps, and the GT labels
        input_feature_split = tf.split(self.input_feat_maps, self.num_gpus)
        input_score_maps_split = tf.split(self.input_score_maps, self.num_gpus)
        input_geo_maps_split = tf.split(self.input_geo_maps, self.num_gpus)
        input_training_masks_split = tf.split(self.input_training_masks, self.num_gpus)
        input_flow_maps_split = tf.split(self.input_flow_maps, self.num_gpus)

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
        # #>>>>>>>>>>>>>>>>>>>>>>>>>>>> collect gradients from different devices and get the averaged gradient, large batch size
        # self.summary_op = tf.summary.merge_all()
        # save moving average
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     FLAGS.moving_average_decay, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())
        # # batch norm updates
        # with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        #     self.train_op = tf.no_op(name='train_op')
        # self.summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())


    def config_train_ops(self):
        # this is used for training with single GPU without regularization
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=5000, decay_rate=0.5, staircase=True)
        self.optim = tf.train.AdamOptimizer(learning_rate)
        self.model_loss = loss(self.score_maps, self.f_score, self.input_geo_maps, self.input_training_masks)
        self.total_loss = tf.add_n([self.model_loss])
        # apply graident
        batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
        if FALGS.apply_grad is 'top':
            tvar1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tiny_embed')
            tvar2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred_module')
            tvars = tvar1 + tvar2
            grads = self.optim.compute_gradients(self.total_loss, tvars)
            apply_gradient_op = self.optim.apply_gradients(grads, global_step=global_step)
        else:
            grads = self.opt.compute_gradients(self.total_loss)
            apply_gradient_op = self.optim.apply_gradients(grads, global_step=global_step)
        # save moving average
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        # batch norm updates
        with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
            self.train_op = tf.no_op(name='train_op')
        # summary writter
        tf.summary.scalar('learning_rate', learning_rate)
        # GT & Loss
        tf.summary.image('score_map', self.input_score_maps)
        tf.summary.image('geo_map_0', self.input_geo_maps[:, :, :, 0:1])
        tf.summary.image('training_masks', self.input_training_masks)
        tf.summary.scalar('model_loss', self.model_loss)
        tf.summary.scalar('total_loss', self.total_loss)
        self.summary_op = tf.summary.merge_all()

    def set_output_tnsrs(self):
        """Initialize output tensors
        """
        if self.mode in ['train_noval', 'train_with_val']:
            # self.y_hat_train_tnsr = [self.loss_op, self.metric_op, self.optim_op, self.g_step_inc_op]
            self.y_hat_train_tnsr = [self.train_op, self.total_loss, self.summary_op]
        # add one validation operation
        if self.mode == 'train_with_val':
            # In online evaluation mode, we only care about the average loss and metric for the batch:
            self.y_hat_val_tnsr = [self.loss_op, self.metric_op]

        if self.mode in ['val', 'val_notrain']:
            # In offline evaluation mode, we only care about the individual predictions and metrics:
            self.y_hat_val_tnsr = [self.f_score, self.f_geometry, self.feature_new]
            self.y_hat_test_tnsr = [self.f_score, self.f_geometry, self.feature_wrapped, self.feature_new]
        # for test mode, we'll use this tensor as output
        if self.mode in ['test']:
            self.y_hat_test_tnsr = [self.f_score, self.f_geometry, self.feature_wrapped, self.feature_new]



    def setup_metrics_ops(self):
        pass

    def init_saver(self):
        """Creates a default saver to load/save model checkpoints. Override, if necessary.
        """
        # if self.mode in ['train_noval', 'train_with_val']:
        #     self.saver = BestCheckpointSaver(self.opts['ckpt_dir'], self.name, self.opts['max_to_keep'], maximize=False)
        # else:
        #     self.saver = tf.train.Saver()
        self.saver = tf.train.Saver(tf.global_variables())
        self.summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())


    def save_ckpt(self, ranking_value=0):
        """Save a model checkpoint
        Args:
            ranking_value: The ranking value by which to rank the checkpoint.
        """
        assert(self.mode in ['train_noval', 'train_with_val'])
        if self.opts['verbose']:
            print("Saving model...")

        # save_path = self.saver.save(self.sess, self.opts['ckpt_dir'] + self.name, self.g_step_op)
        save_path = self.saver.save(ranking_value, self.sess, self.g_step_op)

        if self.opts['verbose']:
            if save_path is None:
                msg = f"... model wasn't saved -- its score ({ranking_value:.2f}) doesn't outperform other checkpoints"
            else:
                msg = f"... model saved in {save_path}"
            print(msg)

    def load_ckpt(self):
        """Load a model checkpoint
        In train mode, load the latest checkpoint from the checkpoint folder if it exists; otherwise, run initializer.
        In other modes, load from the specified checkpoint file.
        """
        if self.mode in ['train_noval', 'train_with_val']:
            if self.opts['train_mode'] == 'fine-tune':
                self.last_ckpt = FLAGS.prev_checkpoint_path
                # In fine-tuning mode, we just want to load the trained params from the file and that's it...
                assert(tf.train.checkpoint_exists(FLAGS.prev_checkpoint_path))
                if self.opts['verbose']:
                    print(f"Initializing from pre-trained model at {FLAGS.prev_checkpoint_path} for finetuning...\n")
                # ...however, the AdamOptimizer also stores variables in the graph, so reinitialize them as well
                # Now initialize the trained params with actual values from the checkpoint
                self.saver.restore(self.sess, FLAGS.prev_checkpoint_path)
                if self.opts['verbose']:
                    print("... model initialized")
            else:
                # In training mode, we either want to start a new training session or resume from a previous checkpoint
                self.last_ckpt = self.saver.best_checkpoint(FLAGS.prev_checkpoint_path , maximize=False)
                if self.last_ckpt is None:
                    self.last_ckpt = tf.train.latest_checkpoint(FLAGS.prev_checkpoint_path)
                # decide whether it's None
                if self.last_ckpt:
                    # We're resuming a session -> initialize the graph with the content of the checkpoint
                    if self.opts['verbose']:
                        print(f"Initializing model from previous checkpoint {self.last_ckpt} to resume training...\n")
                    self.saver.restore(self.sess, self.last_ckpt)
                    if self.opts['verbose']:
                        print("... model initialized")
                else:
                    # Initialize all the variables of the graph from scratch, then assign pre-trained weights
                    if self.opts['verbose']:
                        print(f"Initializing model with random values for initial training...\n")
                    assert (self.mode in ['train_noval', 'train_with_val'])
                    self.sess.run(tf.global_variables_initializer())
                    if self.opts['verbose']:
                        print("... model initialized")
                    # further initialize the weights of detection head
                    reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.pretrained_model_path)
                    tensor_names=['feature_fusion/Conv_7/weights', 'feature_fusion/Conv_7/biases', 'feature_fusion/Conv_8/weights', 'feature_fusion/Conv_8/biases',
                                  'feature_fusion/Conv_9/weights', 'feature_fusion/Conv_9/biases']
                    variable_names = ['pred_module/Conv/weights', 'pred_module/Conv/biases', 'pred_module/Conv_1/weights', 'pred_module/Conv_1/biases',
                                      'pred_module/Conv_2/weights', 'pred_module/Conv_2/biases']
                    # initialize the PWC-flow graph and weights here
                    for t in range(len(variable_names)):
                        wt = reader.get_tensor(tensor_names[t]) # numpy array
                    # get the variables, or related rensors
                        v1 = [var for var in tf.trainable_variables() if var.op.name==variable_names[t]]
                    # tf.assign(v1[0], w1) # won't work because you will add ops to the graph
                        v1[0].load(wt, self.sess)
        # During test, we just need to assign a checkpoint to the model
        else:
            self.last_ckpt = FLAGS.prev_checkpoint_path
            assert(self.last_ckpt is not None)
            if self.opts['verbose']:
                print(f"Loading model checkpoint {self.last_ckpt} for eval or testing...\n")
            self.saver.restore(self.sess, self.last_ckpt)
            if self.opts['verbose']:
                print("... model loaded")


def tower_forward(feature, flow_maps, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        # the core part of the model
        batch_size = FLAGS.batch_size_per_gpu*FLAGS.seq_len
        c = np.zeros((batch_size, 1), dtype=np.int32)
        num = range(2, batch_size, FLAGS.seq_len)
        for i in range(FLAGS.batch_size_per_gpu):
            for j in range(FLAGS.seq_len):
                c[i*5 + j] = num[i]
        indices = tf.constant(c)
        feature = self.input_feat_maps
        flow_maps = self.input_flow_maps
        feature_midframe = tf.manip.gather_nd(feature, indices)
        # create a replicate of al center frames
        K = 2 # range of prev/after frames
        L = 5 # len of video seq
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            tiny_embed = tf.make_template('tiny_embed', embedding_net)
            # [batch_size*5, H, W, C]
            feature_w = dense_image_warp(feature, flow_maps)
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
                self.f_score, self.f_geometry = detector_top(feature_fused)
            self.feature_wrapped = feature_w
            self.feature_new = feature_fused


# scale_by_y2 = tf.make_template('scale_by_y', my_op, scalar_name='y')
def tower_loss(feature, flow_maps, score_maps, geo_maps, training_masks, reuse_variables=None):
    """
    Loss computation on a single GP, which is under the Multi-GPU training strategy:
    First split images/data to gpu first, then construct the model and loss;
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        # the core part of the model
        batch_size = FLAGS.batch_size_per_gpu*FLAGS.seq_len
        c = np.zeros((batch_size, 1), dtype=np.int32)
        num = range(2, FLAGS.batch_size_per_gpu*FLAGS.seq_len, FLAGS.seq_len)
        for i in range(FLAGS.batch_size_per_gpu):
            for j in range(FLAGS.seq_len):
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
            model_loss = loss(score_maps, f_score, geo_maps, f_geometry,training_masks)
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

    ICDAR2013 = '/media/dragonx/DataLight/ICDAR2013/'
    ARC = '/media/dragonx/DataStorage/ARC/'
    if platform.uname()[1] != 'dragonx-H97N-WIFI':
        print("Now it knows it's in a remote cluster")
        ARC = '/work/cascades/lxiaol9/ARC/'
        ICDAR2013 = '/work/cascades/lxiaol9/ARC/EAST/data/ICDAR2013/'

    east_opts = {
    'verbose': True,
    'ckpt_path': ARC + "EAST/checkpoints/east/20180921-173054/" + 'model.ckpt-56092',
    'batch_size': 40,
    'gpu_devices': ['/device:GPU:0', '/device:GPU:1'],
    # controller device to put the model's variables on (usually, /cpu:0 or /gpu:0 -> try both!)
    'controller': '/device:CPU:0',
    'x_dtype': tf.float32,  # image pairs input type
    'x_shape': [512, 512, 3],  # image pairs input shape [2, H, W, 3]
    'y_score_shape': [128, 128, 1],  # u,v flows output type
    'y_geometry_shape': [128, 128, 5],  # u,v flows output shape [H, W, 2]
    'x_mask_shape': [128, 128, 1]
    }
    east_net = EAST(mode='test', options=east_opts)
