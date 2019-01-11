a# !/usr/local/bin
import cv2
import os
import _init_paths
from time import time
from random import randint
import multiprocessing
import numpy as np
from abc import abstractmethod
from scipy import ndimage as nd
from skimage.morphology import watershed, square
import tensorflow as tf
import tqdm
from icdar_smart import load_annotations_solo, check_and_validate_polys, generate_rbox
from tensorpack import DataFlow, DataFromGenerator
from tensorpack.dataflow.parallel import PlasmaGetData, PlasmaPutData  # noqa
from tensorpack import ModelDesc
from tensorpack.dataflow import AugmentImageComponent, BatchData, MultiThreadMapData, PrefetchDataZMQ, MultiProcessMapDataZMQ, dataset, imgaug
from tensorpack.input_source import QueueInput, StagingInput
from tensorpack.models import regularize_cost
from tensorpack.predict import FeedfreePredictor, PredictConfig
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.utils.stats import RatioCounter
# a DataFlow you implement to produce [tensor1, tensor2, ..] lists from whatever sources:

def crop_all_random_seq(num_steps, data, score_maps, geo_maps, training_masks, crop_background=False, max_tries=20):
    '''
    make random crop from the input image
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:QueueInput
    :return:
    '''
    # crop and assemble
    score_map = score_maps[0, :, :]
    flag = False # indication of success
    input_size = 512
    px_size = 512 # real patch size
    py_size = 512 # real patch size
    frame_height, frame_width = score_map.shape
    if frame_height < 512:
        # for i in range(max_tries*15):
        images_new = np.zeros((num_steps, input_size, input_size, 3), dtype=np.uint8)
        scores_new = np.zeros((num_steps, input_size, input_size), dtype=np.uint8)
        geos_new = np.zeros((num_steps, input_size, input_size, 5), dtype=np.float32)
        tmasks_new = np.ones((num_steps, input_size, input_size), dtype=np.uint8)
        x = 0
        y = randint(2, frame_width-py_size-5)
            # print("Iterating over max_tries*15, {}".format(i))
            # if (sum(sum(score_map[:, y-2:y+3])) + sum(sum(score_map[:, y+py_size-2:y+py_size+3]))) == 0:
        flag = True
        images_new[:, :frame_height, :py_size, :] = data[:, :, y:y+py_size, :]
        scores_new[:, :frame_height, :py_size] = score_maps[:, :, y:y+py_size]
        geos_new[:, :frame_height, :py_size, :]= geo_maps[:, :, y:y+py_size, :]
        tmasks_new[:, :frame_height, :py_size] = training_masks[:, :, y:y+py_size]
        return images_new, scores_new, geos_new, tmasks_new
    else:
        new_h, new_w = frame_height, frame_width
        attempt_cnt = 0
        while attempt_cnt<max_tries:
            py_size = 512
            x = randint(2, new_h-px_size-5)
            y = randint(2, new_w-py_size-5)
            if score_map[x, y] > 0:
                continue
            attempt_cnt +=1
            return data[:, x:x+px_size, y:y+py_size, :], score_maps[:, x:x+px_size, y:y+py_size], geo_maps[:, x:x+px_size, y:y+py_size],  training_masks[:, x:x+px_size, y:y+py_size]
    # print('Cropping failed, change the strategy!!!')
    return None, None, None, None

# preprocessing function
class data_raw():
    def __init__(self, video_dir, datapre_path, is_print=False):
        self.video_set = []
        self.video_dir = video_dir
        self.datapre_path = datapre_path
        for root, dirs, files in os.walk(video_dir):
            for file in files:
                if file.endswith('.mp4'):
                    cap = cv2.VideoCapture(video_dir+file)
                    b = {}
                    b["video_name"]   = os.path.splitext(file)[0]
                    b["frame_width"]  = int(cap.get(3))
                    b["frame_height"] = int(cap.get(4))
                    b["frame_num"]    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    xml_solo_path = video_dir + b["video_name"]
                    polys_array_list, tags_array_list, _, _ = load_annotations_solo(xml_solo_path, 1,\
                                                                                  b["frame_num"])
                    b["polys_list"]   = polys_array_list
                    b["tags_list"]    = tags_array_list
                    self.video_set.append(b)
                    cap.release()
        self.total_num = len(self.video_set)
        if is_print:
            for i in range(self.total_num):
                print("{} with {} frames of w:{}, h:{}, polys of the first frame: {}, and tags: {}\n".format(
                                                self.video_set[i]["video_name"],
                                                self.video_set[i]["frame_num"],
                                                self.video_set[i]["frame_width"],
                                                self.video_set[i]["frame_height"],
                                                self.video_set[i]["polys_list"][0],
                                                self.video_set[i]["tags_list"][0]
                )
                     )

class MyDataFlow(DataFlow):
    def __init__(self, raw_data, num_steps, is_training):
        super(MyDataFlow, self).__init__()
        self.raw_data = raw_data
        self.num_steps = num_steps
        self.is_training = is_training
    def __iter__(self):
        raw_data =  self.raw_data
        datapre_path = raw_data.datapre_path
        num_steps   = self.num_steps
        is_training = self.is_training
        if is_training:
            for k in range(240):
                i = randint(0, raw_data.total_num-3-1)
                # for debuging locally
                # if raw_data.video_set[i]["video_name"] not in ["Video_18_3_1", "Video_21_5_1", "Video_26_5_2"]:
                #     continue
                j = randint(0, raw_data.video_set[i]["frame_num"] - num_steps -1)
                new_h, new_w = raw_data.video_set[i]["frame_height"], raw_data.video_set[i]["frame_width"]
                # pre_data
                images = np.zeros([num_steps, new_h, new_w, 3], dtype=np.uint8)
                score_maps = np.zeros([num_steps, new_h, new_w], dtype=np.uint8)
                geo_maps = np.zeros([num_steps, new_h, new_w, 5], dtype=np.float32)
                training_masks = np.ones([num_steps, new_h, new_w], dtype=np.uint8)
                cap = cv2.VideoCapture(raw_data.video_dir+ raw_data.video_set[i]["video_name"] + '.mp4')
                for m in range(num_steps):
                    cap.set(1, j+m)
                    ret, image = cap.read()
                    text_polys, text_tags = raw_data.video_set[i]["polys_list"][j+m], \
                                            raw_data.video_set[i]["tags_list"][j+m]
                    text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (new_h, new_w))
                    if text_polys.shape[0] == 0:# means no boxes here
                        images[m, :, :, :] = image
                        continue
                    # Option 1 read from raw data
                    # score_map_raw, geo_map_raw, training_mask_raw = generate_rbox((new_h, new_w), text_polys, text_tags)
                    # images[m, :, :, :] = image
                    # score_maps[m, :, :] = score_map_raw
                    # geo_maps[m, :, :, :] = geo_map_raw
                    # training_masks[m, :, :] = training_mask_raw
                    # load pre-processed data
                    score_name = datapre_path + 'score_maps/' + '{}'.format(raw_data.video_set[i]["video_name"])+ '/frame'+'{:04d}'.format(j+m)+'.npy'
                    geo_name = datapre_path + 'geo_maps/' + '{}'.format(raw_data.video_set[i]["video_name"])+'/frame'+'{:04d}'.format(j+m)+'.npy'
                    mask_name = datapre_path + 'training_masks/' + '{}'.format(raw_data.video_set[i]["video_name"])+'/frame'+'{:04d}'.format(j+m)+'.npy'
                    images[m, :, :, :]    = image
                    score_maps[m, :, :] = np.load(score_name)
                    geo_maps[m, :, :, :] = np.load(geo_name)
                    training_masks[m, :, :] = np.load(mask_name)
                cap.release()
                imgs_c, score_maps_c, geo_maps_c, training_masks_c = crop_all_random_seq(num_steps, images, score_maps, geo_maps, training_masks)
#                 if if imgs_c is not None
                scores_new = np.copy(score_maps_c)
                for d in range(scores_new.shape[0]):
                    if (sum(scores_new[d, 0, :]) + sum(scores_new[d, 511, :]) + sum(scores_new[d, :, 0]) + sum(scores_new[d, :, 511]))>0:
                        seed_array = np.copy(scores_new[d, :, :])
                        example_array = np.copy(scores_new[d, :, :])
                        distance = nd.distance_transform_edt(example_array)
                        seed_array[1:511, 1:511] = np.zeros((510, 510))
                        result = watershed(-distance, seed_array, mask=example_array, \
                                           connectivity=square(3))
                        # print(type(result[0, 0]))
                        training_masks_c[d, :, :] = training_masks_c[d, :, :]*(1 - result)
                        # training_masks_c[d, :, :] = training_masks_c[d, :, :]*(np.uint8(result) - 254)
                yield [imgs_c, score_maps_c, geo_maps_c, training_masks_c]

def preprocess(dp):
    """
    it will be used something like this
    ds = MultiThreadMapData(ds, 5, preprocess)
    ds = MultiProcessMapDataZMQ(ds, 10, preprocess)
    """
    imgs_c, score_maps_c, geo_maps_c, training_masks_c = dp
    scores_new = np.copy(score_maps_c)
    for d in range(scores_new.shape[0]):
        if (sum(scores_new[d, 0, :]) + sum(scores_new[d, 511, :]) + sum(scores_new[d, :, 0]) + sum(scores_new[d, :, 511]))>0:
            seed_array = np.copy(scores_new[d, :, :])
            example_array = np.copy(scores_new[d, :, :])
            distance = nd.distance_transform_edt(example_array)
            seed_array[1:511, 1:511] = np.zeros((510, 510))
            result = watershed(-distance, seed_array, mask=example_array, \
                               connectivity=square(3))
            print(type(result[0, 0]))
            # result_reverse = 1 - result
            # training_masks_c[d, :, :] = training_masks_c[d, :, :]*result_reverse
            training_masks_c[d, :, :] = training_masks_c[d, :, :]*(1 - result)
    return [imgs_c, score_maps_c, geo_maps_c, training_masks_c]



if __name__ == "__main__":
    #
    import matplotlib.pyplot as plt
    ICDAR2013 = '/media/lxiaol9/DataStorage/ARC/EAST/data/ICDAR2013/'
    video_dir = ICDAR2013+'/train/'
    # we may not need to process
    data_dir = '/media/lxiaol9/DataStorage/ARC/EAST/data/pre-processed/'
    t1_start = time()
    dr = data_raw(video_dir, data_dir, is_print=True)
    df = MyDataFlow(raw_data=dr, num_steps=5, is_training=True)
    # df = MultiProcessMapDataZMQ(df, 4, preprocess)
    df = BatchData(df, 2)
    # df = PrefetchDataZMQ(df, 2)
    # df = PlasmaGetData(df)
    df.reset_state()
    t1_end = time()
    print("data loader preparation costs {} seconds".format(t1_end - t1_start))
    step = 1
    for datapoint in df:
        # vis
        for j in range(len(datapoint)):
            print(datapoint[j].shape)
        # import pdb;pdb.set_trace()
        nrows, ncols=2, 4
        heights = [50 for a in range(nrows)]
        widths = [50 for a in range(ncols)]
        cmaps = [['viridis', 'binary'], ['plasma', 'coolwarm'], ['Greens', 'copper']]
        fig_width = 10  # inches
        fig_height = fig_width * sum(heights) / sum(widths)
        fig,axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios':heights}) #define to be 2 rows, and 4cols.
        image = np.arange(3000).reshape((50,60))
        for i in range(nrows):
            # [datatype, batch_size, num_steps, demensions]
            axes[i, 0].imshow(datapoint[0][i][0, :, :, :])
            axes[i, 1].imshow(datapoint[1][i][0, :, :]*255)
            axes[i, 2].imshow(datapoint[2][i][0, :, :, 0])
            axes[i, 3].imshow(datapoint[3][i][0, :, :])
            for j in range(ncols):
                axes[i, j].axis('off')
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0.1, wspace = 0.1)
        plt.show()
        step = step+1
        fig.savefig(ICDAR2013+'vis/{}'.format(step)+".png")
        print("now passed {} seconds".format(time() - t1_end))
