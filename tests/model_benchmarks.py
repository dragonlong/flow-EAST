##===================== Statements & Copyright ===================##
"""
LOG:     April 12th
AUTHOR:  Xiaolong Li, VT
CONTENT: Used for Video-text project, reuse the LSTM model
Logs: when using this LSTM model, we need to take care of the 3 models' relationship
Model class is defined in the VectModel, with LSTM and its output, loss, defined, the
model summay is added in the step of object instaniation step;

train contains the graph of LSTM, while val and test reuse the variables but don't
have the part of op_eval
"""
# Demo on using LSTM, tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime
from random import randint
from matplotlib import pyplot as plt
from lstm.rnn_eval import model_eval
from lstm.configrnn import get_config
from lstm.modelrnn_new import VectModel
from lstm.input_node import DetectorInputMul
from tensorflow.python.client import device_lib
from model_zoo import draw_illu, convert_vect
############ Macros ############
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION = "unidirection"

###############################
# FLAGS or args.parser
#####################################################
flags = tf.app.flags
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
now = datetime.now()
# The only part you need to modify during training
flags.DEFINE_string("model", "test", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", "/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process", "Where data is stored" )
flags.DEFINE_string("save_path", "/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/LSTM/"+now.strftime("%Y%m%d-%H%M%S"), "Model output")
flags.DEFINE_string("vis_path", "/media/dragonx/DataStorage/ARC/EASTRNN/vis/LSTM/"+now.strftime("%Y%m%d-%H%M%S"), "save visualization")
flags.DEFINE_integer("num_gpus", 1, "Larger than 1 will create multiple training replicas")
flags.DEFINE_string("rnn_mode", CUDNN, "one of CUDNN: BASIC, BLOCK")
flags.DEFINE_boolean("random", True, "style when feeding grouped frames data")
flags.DEFINE_boolean("source", True, "whether load data from source")
flags.DEFINE_boolean("dis_plt", False, "whether using pyplot real-time display ")
flags.DEFINE_boolean("img_save", True, "whether we need to save the visualization")
flags.DEFINE_boolean("fine_tune", True, "whether to train from pre-trained model")
flags.DEFINE_integer('dis_freq', 1, "frequency on training epoch when visualization")
flags.DEFINE_integer('interest_num', 50, 'number of images to test')


def main(_):
    FLAGS = flags.FLAGS
    if not FLAGS.data_path:
        raise ValueError("Must set --")
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if FLAGS.num_gpus > len(gpus):
        raise ValueError("Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))
    config = get_config(FLAGS)
    eval_config = get_config(FLAGS)
    eval_config.batch_size = 1
    eval_config.num_steps = 1
    eval_config.epoch_size = 50
    with tf.Graph().as_default():
        # Global initializer for Variables in the model
        initializer = tf.random_normal_initializer()
        # Construct the model graph
        with tf.name_scope("Train"):
            # use placeholder to stand for input and targets
            x_train = tf.placeholder(tf.float32, shape=[None, config.num_steps, config.vocab_size])
            y_train = tf.placeholder(tf.float32, shape=[None, config.num_steps, config.output_size])
            with tf.variable_scope("Model", reuse = None, initializer=initializer):
                m = VectModel(True, config, x_train, y_train)
            training_cost_sum = tf.summary.scalar("Loss", m.cost)
            training_lr = tf.summary.scalar("Learning_Rate", m.lr)
            m.summary_merged = tf.summary.merge([training_lr, training_cost_sum])
        with tf.name_scope("Test"):
            x_test = tf.placeholder(tf.float32, shape=[None, eval_config.num_steps, eval_config.vocab_size])
            y_test = tf.placeholder(tf.float32, shape=[None, eval_config.num_steps, eval_config.output_size])
            with tf.variable_scope("Model", reuse =True, initializer=initializer):
                mtest = VectModel(False, eval_config, x_test, y_test)
            test_cost_sum = tf.summary.scalar("Loss", mtest.cost)
            mtest.summary_merged = test_cost_sum

        # A training helper that checkpoints models and computes summaries.
        # ======================== initialize from the saved weights ============================#
        checkpoint_path = "/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/ARC/checkpoints/LSTM/"
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(
                'Checkpoint `{}` not found'.format(checkpoint_path))
        saver = tf.train.Saver()
        # restore the model from weights
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        model_path = os.path.join(checkpoint_path, '20180411-174228-69600')
        logger.info('Restore from {}'.format(model_path))
        saver.restore(session, model_path)
        ################### load  all data into memory ###################
        datapath = FLAGS.data_path
        input = DetectorInputMul(datapath, 1, 12)
        ################### choose video and frame to test ###############
        if FLAGS.random is True:
            i = randint(0, len(input.cnt_frame) - 1)
        else:
            i = 1
        print('choosing No. %d video, with shape [%d, %d]' % (i, input.data[0].shape[0], input.data[i].shape[1]))
        feed_dict = {}
        _, l1 = input.data[i].shape
        _, l2 = input.targets[0].shape
        data = np.zeros([eval_config.batch_size, eval_config.num_steps, l1])
        targets = np.zeros([eval_config.batch_size, eval_config.num_steps, l2])
        frame_set = []
        for m in range(FLAGS.interest_num):
            if FLAGS.random is True:
                i = randint(0, len(input.cnt_frame) - 1)
            else:
                i = m
            for k in range(eval_config.batch_size):
                j = randint(0, input.cnt_frame[i] - eval_config.num_steps)
                # print('choosing starting frame %d, with num_steps is %d' % (j, config.num_steps))
                frame_set.append(j / input.cnt_frame[i])
                data[k, :, :] = input.data[i][j:(j + eval_config.num_steps), :]
                targets[k, :, :] = input.targets[i][j:(j + eval_config.num_steps), :]
            feed_dict[mtest.input_data] = data
            feed_dict[mtest.targets] = targets
            state = session.run(mtest.initial_state)
            for i, (c, h) in enumerate(mtest.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
            predicts = session.run(mtest.prediction, feed_dict)
            precision, recall, f1 = model_eval(targets, predicts, input.video_name, i, frame_set)
            base_path = FLAGS.data_path
            video_path = base_path + '/' + input.video_name[i] + '.avi'
            # read the video frame and related infos
            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            cap.set(2, frame_set[0])
            ret, frame = cap.read()
            # draw a new box on top of the current frame
            predict = convert_vect(predicts[0, 0, :], frame_width, frame_height)
            newimage = draw_illu(frame.copy(), predict)
            # get heatmap from data
            heatmap = np.reshape(data[0, 0, :], (160, 160))
            fig1 = plt.figure(figsize=(20, 10))
            fig1.add_subplot(1, 2, 1)
            plt.imshow(newimage)
            plt.title("Ground Truth & EAST Prediction")
            fig1.add_subplot(1, 2, 2)
            plt.imshow(heatmap)
            plt.title('Input heat map')
            if FLAGS.dis_plt is True:
                plt.show()
            if FLAGS.img_save is True:
                fig1.savefig(FLAGS.vis_path + "/"+input.video_name[i] + ".png")


if __name__ == '__main__':
    tf.app.run()
