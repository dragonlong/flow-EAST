import tensorflow as tf
import numpy as np

from tensorflow.contrib import slim

# tf.app.flags.DEFINE_integer('text_scale', 512, '')
import platform
import _init_paths
from nets import resnet_v1

FLAGS = tf.app.flags.FLAGS


def unpool(inputs):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*2,  tf.shape(inputs)[2]*2])


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


class EAST(object):
    """
    A self-contained module for all use
    """
    def __init__(self, name='base', mode=None, options=None, session=None):
        # Class Attributes
        assert(mode in ['train_noval', 'train_with_val', 'val', 'val_notrain', 'test'])
        self.mode, self.opts, self.sess= mode, options, session
        self.y_hat_train_tensor = self.y_hat_val_tensor = self.y_hat_test_tensor = None
        self.name = name
        self.num_gpus = len(self.opts['gpu_devices'])
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
        # default 1 gpu
        elif self.mode in ['val', 'val_notrain']:
            self.build_model()
            self.setup_metrics_ops()

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
        batch_size = self.num_gpus * self.opts['batch_size_per_gpu']
        self.x_tnsr = tf.placeholder(tf.float32, [batch_size] + self.opts['x_shape'], 'x_tnsr')
        self.y_score = tf.placeholder(tf.float32, [batch_size] + self.opts['y_score_shape'], 'input_score_maps')
        self.y_geometry = tf.placeholder(tf.float32, [batch_size] + self.opts['y_geometry_shape'], name='input_geo_maps')
        self.x_mask = tf.placeholder(tf.float32, [batch_size] + self.opts['x_mask_shape'], name='input_training_masks')


    # Necessary methods for public usage
    def build_model(self):
        self.f_score, self.f_geometry, self.feature = model(self.x_tnsr, is_testing=False)


    def build_model_towers(self):
        # split only among the features, flow maps, and the GT labels
        input_images_split = tf.split(self.x_tnsr, self.num_gpus)
        tower_features = []
        reuse_variables = None
        for i, gpu_id in enumerate(self.opts['gpu_devices']):
            with tf.device(gpu_id):
                with tf.name_scope('model_%d' % i) as scope:
                    iis = input_images_split[i]
                    # create separate graph in different device
                    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                        _, _, feature = model(iis, is_testing=False)
                    tower_features.append(feature)
                    reuse_variables = True
        self.tower_features = tf.concat(tower_features, 0)


    def build_model_towers_loss(self):
        # split only among the features, flow maps, and the GT labels
        # input_images_split = tf.split(input_images, len(gpus))
        # input_score_maps_split = tf.split(input_score_maps, len(gpus))
        # input_geo_maps_split = tf.split(input_geo_maps, len(gpus))
        # input_training_masks_split = tf.split(input_training_masks, len(gpus))
        # input_flow_maps_split = tf.split(input_flow_maps, len(gpus))
        #
        # tower_grads = []
        # reuse_variables = None
        # tvars = []
        # for i, gpu_id in enumerate(gpus):
        #     with tf.device('/gpu:%d' % gpu_id):
        #         with tf.name_scope('model_%d' % gpu_id) as scope:
        #             iis = input_images_split[i]
        #             ifms = input_flow_maps_split[i]
        #             isms = input_score_maps_split[i]
        #             igms = input_geo_maps_split[i]
        #             itms = input_training_masks_split[i]
        #             # create separate graph in different device
        #             total_loss, model_loss = tower_loss(iis, ifms, isms, igms, itms, reuse_variables)
        #             batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
        #             reuse_variables = True
        #             tvar1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tiny_embed')
        #             tvar2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pred_module')
        #             tvars = tvar1 + tvar2
        #             grads = opt.compute_gradients(total_loss, tvars)
        #             tower_grads.append(grads)
        #
        # grads = average_gradients(tower_grads)
        # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        pass

    def set_output_tnsrs(self):
        """Initialize output tensors
        """
        if self.mode in ['train_noval', 'train_with_val']:
            # self.y_hat_train_tnsr = [self.loss_op, self.metric_op, self.optim_op, self.g_step_inc_op]
            self.y_hat_train_tnsr = [self.loss_op, self.optim_op]

        if self.mode == 'train_with_val':
            # In online evaluation mode, we only care about the average loss and metric for the batch:
            self.y_hat_val_tnsr = [self.loss_op]

        if self.mode in ['val', 'val_notrain']:
            # In offline evaluation mode, we only care about the individual predictions and metrics:
            self.y_hat_val_tnsr = [self.f_score, self.f_geometry, self.feature]
            # if self.opts['sparse_gt_flow'] is True:
            #     # Find the location of the zerod-out flows in the gt
            #     zeros_loc = tf.logical_and(tf.equal(self.y_tnsr[:, :, :, 0], 0.0), tf.equal(self.y_tnsr[:, :, :, 1], 0.0))
            #     zeros_loc = tf.expand_dims(zeros_loc, -1)
            #
            #     # Zero out flow predictions at the same location so we only compute the EPE at the sparse flow points
            #     sparse_flow_pred_tnsr = tf_where(zeros_loc, tf.zeros_like(self.flow_pred_tnsr), self.flow_pred_tnsr)
            #
            #     self.y_hat_val_tnsr = [sparse_flow_pred_tnsr, self.metric_op]
        if self.num_gpus==1:
            self.y_hat_test_tnsr = [self.feature]
        else:
            self.y_hat_test_tnsr = [self.tower_features]

    def config_train_ops(self):
        pass

    def setup_metrics_ops(self):
        self.metric_op = None

    def init_saver(self):
        """Creates a default saver to load/save model checkpoints. Override, if necessary.
        """
        if self.mode in ['train_noval', 'train_with_val']:
            self.saver = BestCheckpointSaver(self.opts['ckpt_dir'], self.name, self.opts['max_to_keep'], maximize=False)
        else:
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            self.saver = tf.train.Saver(variable_averages.variables_to_restore())


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
            self.last_ckpt = None
            if self.opts['train_mode'] == 'fine-tune':
                # In fine-tuning mode, we just want to load the trained params from the file and that's it...
                assert(tf.train.checkpoint_exists(self.opts['ckpt_path']))
                if self.opts['verbose']:
                    print(f"Initializing from pre-trained model at {self.opts['ckpt_path']} for finetuning...\n")
                # ...however, the AdamOptimizer also stores variables in the graph, so reinitialize them as well
                self.sess.run(tf.variables_initializer(self.optim.variables()))
                # Now initialize the trained params with actual values from the checkpoint
                _saver = tf.train.Saver(var_list=tf.trainable_variables())
                _saver.restore(self.sess, self.opts['ckpt_path'])
                if self.opts['verbose']:
                    print("... model initialized")
                self.last_ckpt = self.opts['ckpt_path']
            else:
                # In training mode, we either want to start a new training session or resume from a previous checkpoint
                self.last_ckpt = self.saver.best_checkpoint(self.opts['ckpt_dir'], maximize=False)
                if self.last_ckpt is None:
                    self.last_ckpt = tf.train.latest_checkpoint(self.opts['ckpt_dir'])

                if self.last_ckpt:
                    # We're resuming a session -> initialize the graph with the content of the checkpoint
                    if self.opts['verbose']:
                        print(f"Initializing model from previous checkpoint {self.last_ckpt} to resume training...\n")
                    self.saver.restore(self.sess, self.last_ckpt)
                    if self.opts['verbose']:
                        print("... model initialized")
                else:
                    # Initialize all the variables of the graph
                    if self.opts['verbose']:
                        print(f"Initializing model with random values for initial training...\n")
                    assert (self.mode in ['train_noval', 'train_with_val'])
                    self.sess.run(tf.global_variables_initializer())
                    if self.opts['verbose']:
                        print("... model initialized")
        else:
            # Initialize the graph with the content of the checkpoint
            self.last_ckpt = self.opts['ckpt_path']
            assert(self.last_ckpt is not None)
            if self.opts['verbose']:
                print(f"Loading model checkpoint {self.last_ckpt} for eval or testing...\n")
            self.saver.restore(self.sess, self.last_ckpt)
            if self.opts['verbose']:
                print("... model loaded")


def model(images, weight_decay=1e-5, is_training=True, is_testing=True):
    '''
    define the model, we use slim's implemention of resnet
    '''
    images = mean_image_subtraction(images)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=weight_decay)):
        logits, end_points = resnet_v1.resnet_v1_50(images, is_training=is_training, scope='resnet_v1_50')

    with tf.variable_scope('feature_fusion', values=[end_points.values]):
        batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
        }
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            f = [end_points['pool5'], end_points['pool4'],
                 end_points['pool3'], end_points['pool2']]
            for i in range(4):
                print('Shape of f_{} {}'.format(i, f[i].shape))
            g = [None, None, None, None]
            h = [None, None, None, None]
            num_outputs = [None, 128, 64, 32]
            for i in range(4):
                if i == 0:
                    h[i] = f[i]
                else:
                    c1_1 = slim.conv2d(tf.concat([g[i-1], f[i]], axis=-1), num_outputs[i], 1)
                    h[i] = slim.conv2d(c1_1, num_outputs[i], 3)
                if i <= 2:
                    g[i] = unpool(h[i])
                else:
                    g[i] = slim.conv2d(h[i], num_outputs[i], 3)
                print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))
            # here we use a slightly different way for regression part,
            # we first use a sigmoid to limit the regression range, and also
            # this is do with the angle map
            if is_testing:
                return g[3]
            else:
                F_score = slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)
                # 4 channel of axis aligned bbox and 1 channel rotation angle
                geo_map = slim.conv2d(g[3], 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * FLAGS.text_scale
                angle_map = (slim.conv2d(g[3], 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi/2 # angle is between [-45, 45]
                F_geometry = tf.concat([geo_map, angle_map], axis=-1)
                return F_score, F_geometry, g[3]


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
    'batch_size_per_gpu': 20,
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
