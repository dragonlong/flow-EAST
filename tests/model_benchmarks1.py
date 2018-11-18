##===================== Statements & Copyright ===================##
"""
LOG:     Aug. 19th
AUTHOR:  Xiaolong Li, VT
CONTENT: Used for Video-text project, reuse the 2D Convolutional LSTM model trained for heatmap
Logs:    Here we will choose ArrayModel defined from model_heatmap.py
         also import HeatInputMul from input_node.py
"""
# Demo on using LSTM, tensorflow
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from matplotlib import pyplot as plt
import cv2
#
import io
import os
import platform
import logging
import collections
from datetime import datetime
from random import randint
#
from utils.rnn_eval import model_eval
from utils.rnn_eval import draw_illu
from utils.model_zoo import convert_vect
import utils.util as util
from config.configrnn import get_config
#from lstm.modelrnn_new import VectModel
#from lstm.input_node import DetectorInputMul
from lstm.model_heatmap import ArrayModel
from utils.input_node import DetectorInputMul, HeatInputMul
from tensorflow.python.client import device_lib

############ Macros ############
BASIC = "basic"
CUDNN = "cudnn"
BLOCK = "block"
CONV  = "conv2d"

CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION = "unidirection"

# FLAGS or args.parser
#####################################################
flags = tf.app.flags
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
now = datetime.now()

# The only part you need to modify during training
flags.DEFINE_string("data_path", "/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process", "Where data is stored" )
flags.DEFINE_string("save_path", "/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/LSTM/"+now.strftime("%Y%m%d-%H%M%S"), "Model output")
flags.DEFINE_string("base_path", "/media/dragonx/DataStorage/ARC/EASTRNN/data/GAP_process", "root directory for data")
flags.DEFINE_string("vis_path", "/media/dragonx/DataStorage/ARC/EASTRNN/vis/LSTM/"+now.strftime("%Y%m%d-%H%M%S"), "save visualization")
flags.DEFINE_string("video_path", "/media/dragonx/DataStorage/Video Detection/ICDAR/train/", "where video files are stored")
flags.DEFINE_string('pretrained_model_path', '/media/dragonx/DataStorage/ARC/EASTRNN/weights/EAST/resnet_v1_50.ckpt', '')
flags.DEFINE_string("system", "local", "deciding running env")
flags.DEFINE_string("checkpoints_path", "/home/dragonx/Documents/VideoText2018/EAST-master/weights/east_icdar2015_resnet_v1_50_rbox/", "checkpoints")
# Model running infos
flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
flags.DEFINE_string("model", "test", "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_integer("num_gpus", 1, "Larger than 1 will create multiple training replicas")
flags.DEFINE_string("rnn_mode", CONV, "one of CUDNN: BASIC, BLOCK")
flags.DEFINE_boolean("random", True, "style when feeding grouped frames data")
flags.DEFINE_boolean("source", False, "whether load data from source")
# visualization
flags.DEFINE_boolean("dis_plt", False, "whether using pyplot real-time display ")
flags.DEFINE_boolean("img_save", True, "whether we need to save the visualization")
flags.DEFINE_integer('dis_freq', 1, "frequency on training epoch when visualization")
flags.DEFINE_integer("dis_step", 20, "at No. step to display")
flags.DEFINE_integer("num_readers", 24, "process used to fetch data")
flags.DEFINE_integer('interest_num', 50, 'number of images to test')

FLAGS = flags.FLAGS

def main(_):
    # to increase the code robustness
    if platform.uname()[1] != 'dragonx-H97N-WIFI':
        print("Now it knows it's in a remote cluster")
        FLAGS.system = "remote"
        FLAGS.data_path = "/home/lxiaol9/ARC/EASTRNN/data/GAP_process/"
        FLAGS.vis_path = "/home/lxiaol9/ARC/EASTRNN/vis/LSTM/"
        FLAGS.save_path = "/home/lxiaol9/ARC/EASTRNN/checkpoints/LSTM/" + now.strftime("%Y%m%d-%H%M%S")
        FLAGS.video_path = "/home/lxiaol9/ARC/EASTRNN/data/ICDAR/train/"
        FLAGS.checkpoints_path = "/home/lxiaol9/ARC/EASTRNN/weights/EAST/east_icdar2015_resnet_v1_50_rbox/"
        FLAGS.pretrained_model_path = "/home/lxiaol9/ARC/EASTRNN/weights/EAST/resnet_v1_50.ckpt"
    if not FLAGS.data_path:
        raise ValueError("Must set --")
    print("############## Step1: The environment path has been set up ###############")
    if not FLAGS.data_path:
        raise ValueError("Must set --")
    gpus = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if FLAGS.num_gpus > len(gpus):
        raise ValueError("Your machine has only %d gpus "
        "which is less than the requested --num_gpus=%d."
        % (len(gpus), FLAGS.num_gpus))
    config = get_config(FLAGS)
    config.batch_size = 1
    with tf.Graph().as_default():
        # Global initializer for Variables in the model
        initializer = tf.random_normal_initializer()
        # Construct the model graph
        with tf.name_scope("Train"):
            initializer = tf.random_normal_initializer()
            # use placeholder to stand for input and targets
            x_train = tf.placeholder(tf.float32, shape=[None, config.num_steps, None, None, 3])
            model = ArrayModel(True, config, x_train, reuse_variables=None, initializer=initializer)
        # ======================== initialize from the saved weights ============================#
        if platform.uname()[1] != 'dragonx-H97N-WIFI':
            checkpoint_path = "/home/lxiaol9/ARC/EASTRNN/checkpoints/LSTM/"
        else:
            checkpoint_path = "/media/dragonx/DataStorage/ARC/EASTRNN/checkpoints/ARC/checkpoints/LSTM/"
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(
                'Checkpoint `{}` not found'.format(checkpoint_path))
        saver = tf.train.Saver()
        # restore the model from weights
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
        model_path = os.path.join(checkpoint_path, '20180818-170857-11100')
        logger.info('Restore from {}'.format(model_path))
        saver.restore(session, model_path)
        print("##############Step 2: Weight restoring successfully ################")
        ################### load  all data into memory ###################
        if FLAGS.source is True:
            datapath = FLAGS.data_path
            test_input = HeatInputMul(datapath, 1, 12, 2)
            input = DetectorInputMul(datapath, 1, 12, 1) # datapath, video_start, video_end, dimension
        else:
            datapath = FLAGS.data_path
            test_input = HeatInputMul(datapath, 1, 12, 0)
            input = DetectorInputMul(datapath, 1, 12, 1) # we will use input.targets[videos, frames, vect]
        print("##############Step 3: Heatmap, GT data is ready now################")
        ################### choose video and frame to test ###############
        i = test_input.video_name.index('Video_37_2_3')
        iters = 0.0
        state = session.run(model.initial_state)
        # tensors dict
        fetches = {
                #"cost": m.cost,
                "final_state": model.final_state,
                # "loss": m.loss,
                "heat_map_pred": model.heatmap_predict
                }
        # frames sequence: 0, 1, 2;3, 4, 5;6, 7, 8
        for step in range(int((test_input.cnt_frame[i] - 1)/3)):
            feed_dict = {}
            data = np.zeros([config.batch_size, config.num_steps, config.shape[0], config.shape[1], 3], dtype=np.float32)
            heat_maps = np.zeros([config.batch_size, config.num_steps, int(config.shape[0]/4), int(config.shape[0]/4), 1], dtype=np.float32)
            # randomly choosing starting frame
            frame_set =[]
            video_file = FLAGS.video_path + test_input.video_name[i] + '.mp4'
            gt_path = FLAGS.data_path + test_input.video_name[i]+'/gt/'
            # frame number to choose
            j = step*3
            cap = cv2.VideoCapture(video_file)
            for m in range(config.num_steps):
                cap.set(1, (j+m))
                ret, frame = cap.read()
                data[0, m, :, :, :] = cv2.resize(frame, (config.shape[0], config.shape[1]))
                heat_maps[0, m, :, :, 0] = cv2.resize(np.squeeze(np.load(gt_path+'frame'+'{0:03d}'.format(j+m+1)+'.npy')), (128, 128))
            # print('choosing starting frame %d, with num_steps is %d' % (j, config.num_steps))
            cap.release()
            frame_set.append(j)

            feed_dict[model.input_data] = data
            feed_dict[model.input_heat_maps] = heat_maps
            for i, (c, h) in enumerate(model.initial_state):
                feed_dict[c] = state[i].c
                feed_dict[h] = state[i].h
            vals = session.run(fetches, feed_dict=feed_dict)
            state = vals["final_state"]
            heat_map_pred = vals["heat_map_pred"]
            print(len(heat_map_pred))
            #
            iters += config.num_steps
            npyname1 = FLAGS.vis_path + 'Video_37_2_3/' + 'frame' + format(j, '03d')
            npyname2 = FLAGS.vis_path + 'Video_37_2_3/' + 'frame' + format(j+1, '03d')
            npyname3 = FLAGS.vis_path + 'Video_37_2_3/' + 'frame' + format(j+2, '03d')
            np.save(npyname1, heat_map_pred[0])
            np.save(npyname2, heat_map_pred[1])
            np.save(npyname3, heat_map_pred[2])
            print("saving frame at %d" %(j))
            # heatmap is a list with [0, 128, 128, 0] matrix for 3 frames accroding to our need
            # box regression from heatmap predicts
            # Calculating the p, r, f1

            # precision, recall, f1 = model_eval(targets, predicts, input.video_name, i, frame_set)
            # base_path = FLAGS.data_path
            # video_path = base_path + '/' + input.video_name[i] + '.avi'
            # # read the video frame and related infos
            # cap = cv2.VideoCapture(video_path)
            # frame_width = int(cap.get(3))
            # frame_height = int(cap.get(4))
            # cap.set(2, frame_set[0])
            # ret, frame = cap.read()
            # # draw a new box on top of the current frame
            # predict = convert_vect(predicts[0, 0, :], frame_width, frame_height)
            # newimage = draw_illu(frame.copy(), predict)
            # # get heatmap from data
            # heatmap = np.reshape(data[0, 0, :], (160, 160))
            # fig1 = plt.figure(figsize=(20, 10))
            # fig1.add_subplot(1, 2, 1)
            # plt.imshow(newimage)
            # plt.title("Ground Truth & EAST Prediction")
            # fig1.add_subplot(1, 2, 2)
            # plt.imshow(heatmap)
            # plt.title('Input heat map')
            # if FLAGS.dis_plt is True:
            #     plt.show()
            # if FLAGS.img_save is True:
            #     fig1.savefig(FLAGS.vis_path + "/"+input.video_name[i] + ".png")


if __name__ == '__main__':
    tf.app.run()
