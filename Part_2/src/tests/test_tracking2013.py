"""
Author: Xiaolong Li, Virginia Tech
Log: Oct. 13th, create file, initial debug
"""
from __future__ import print_function

from numba import jit
import platform
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
import cv2
# my own model
import _init_paths
from bayes.sort import motion_bayestrack

idname1 = '20180921-173054'
idname2 = 'model.ckpt-56092'
ICDAR2013 = '/media/dragonx/DataLight/ICDAR2013/'
ARC = '/media/dragonx/DataStorage/ARC/'
idx = 0  # initial frame number
if platform.uname()[1] != 'dragonx-H97N-WIFI':
    print("Now it knows it's in a remote cluster")
    ARC = '/work/cascades/lxiaol9/ARC/'
    ICDAR2013 = '/work/cascades/lxiaol9/ARC/EAST/data/ICDAR2013/'
#>>>>>>>>>>>>>>>>>>>>>>Sort test video>>>>>>>>>>>>>>>>>>>>>>>>>>>#
test_data_path = ICDAR2013+'test/'
checkpoint_path = ARC + 'EAST/checkpoints/east/'
dets_path = ICDAR2013 + 'test_results1/'
out_path = ICDAR2013 + 'tracking_output/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
video_set = ['Video_11_4_1', 'Video_1_1_2', 'Video_17_3_1', 'Video_5_3_2', 'Video_6_3_2']

if __name__ == '__main__':
    sequences = video_set[0:1]
    display = True
    phase = 'train'
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32,3) #used only for display
    if(display):
        # interactive mode on
        plt.ion()
        fig = plt.figure()
    for seq in sequences:
        # initialization
        mot_tracker = motion_bayestrack() #create instance of the SORT tracker
        seq_dets = np.loadtxt(dets_path+'%s.txt'%(seq),delimiter=',') #load detections
        with open(out_path+'%s.txt'%(seq),'w') as out_file:
            print("Processing %s."%(seq))
            # frame info actually comes from the first digit
            video_name = '%stest/%s.mp4'%(ICDAR2013, seq)
            if(display):
                cap = cv2.VideoCapture(video_name)
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                ratio_h, ratio_w = 512/frame_height, 512/frame_width
            for frame in range(90, int(seq_dets[:,0].max())):
                # load the image,
                if(display):
                    cap.set(1, frame)
                    ret, image = cap.read()
                    ax1 = fig.add_subplot(111, aspect='equal')
                    ax1.imshow(image)
                    plt.title(seq+' Tracked Targets')
                dets = seq_dets[seq_dets[:,0]==frame,2:11]#(x_i, y_i, score)
                # print(dets)
                total_frames += 1
                start_time = time.time()
                trackers = mot_tracker.update(dets)
                cycle_time = time.time() - start_time
                total_time += cycle_time
                # save the updated detections
                # print("Trackers are ", trackers)
                for i, box in enumerate(trackers):
                    print("{:d},".format(frame)+',-1,'.join(["{:2.1f}".format(x) for x in box])+',-1,-1,-1\n', file=out_file)
                    if(display):
                        box[0:8:2] = box[0:8:2]/ratio_w
                        box[1:8:2] = box[1:8:2]/ratio_h
                        box= box.astype(np.int32)
                        ax1.add_patch(patches.Polygon(box[0:8].reshape((4,2)),fill=False,lw=2,ec=colours[box[8]%32,:]))
                        ax1.set_adjustable('box-forced')
                fig.savefig(out_path+'frame{}.png'.format(frame))
                if(display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()
            if(display):
                cap.release()

    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
