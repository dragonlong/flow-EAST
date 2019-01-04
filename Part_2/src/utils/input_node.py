import utils.reader as reader
import numpy as np
from scipy import ndimage, misc
import tensorflow as tf



class DetectorInput(object):
    """The input data."""
    def __init__(self, config, frame_start, frame_end, FLAGS, name=None):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.epoch_size = (((frame_end- frame_start) // batch_size) - 1) // num_steps
        self.input_data, self.targets = reader.vect_producer(FLAGS.data_path, frame_start, frame_end, batch_size, num_steps, name=name)


class DetectorInputMul(object):
    """The input data."""
    def __init__(self, datapath, video_start, video_end, dimension):
        if dimension == 1:
            self.data, self.targets, self.cnt_frame, self.video_name = reader.vect_producer_mul(datapath, video_start,  video_end)
        if dimension == 2:
            self.data, self.input_geo_maps, self.input_score_maps,  self.input_training_masks, self.cnt_frame, self.video_name = \
                reader.array_producer_mul(datapath, video_start,  video_end)
        if dimension == 0:
            self.data = []
            self.input_geo_maps = []
            self.input_score_maps = []
            self.input_training_masks = []
            self.cnt_frame = [454, 300, 300, 300]
            self.video_name = ['Video_37_2_3', 'Video_51_7_4', 'Video_33_2_3', 'Video_47_6_4']
            self.data.append(np.load(datapath + '/data.npy'))
            self.input_geo_maps.append(np.load(datapath + '/score.npy'))
            self.input_score_maps.append(np.load(datapath + '/geo.npy'))
            self.input_training_masks.append(np.load(datapath + '/mask.npy'))
            self.data.append(np.load(datapath + '/data1.npy'))
            self.input_geo_maps.append(np.load(datapath + '/score1.npy'))
            self.input_score_maps.append(np.load(datapath + '/geo1.npy'))
            self.input_training_masks.append(np.load(datapath + '/mask1.npy'))
            self.data.append(np.load(datapath + '/data2.npy'))
            self.input_geo_maps.append(np.load(datapath + '/score2.npy'))
            self.input_score_maps.append(np.load(datapath + '/geo2.npy'))
            self.input_training_masks.append(np.load(datapath + '/mask2.npy'))
            self.data.append( np.load(datapath + '/data3.npy'))
            self.input_geo_maps.append( np.load(datapath + '/score3.npy'))
            self.input_score_maps.append(np.load(datapath + '/geo3.npy'))
            self.input_training_masks.append( np.load(datapath + '/mask3.npy'))


class VideoInputMul(object):
    """The input data."""
    def __init__(self, datapath, video_start, video_end, dimension):
        if dimension == 1:
            _, self.targets, self.cnt_frame, self.video_name = reader.vect_producer_mul(datapath, video_start,  video_end)
        if dimension == 2:
            _, self.input_geo_maps, self.input_score_maps,  self.input_training_masks, self.cnt_frame, self.video_name = \
                reader.array_producer_mul(datapath, video_start,  video_end)
        if dimension == 0:
            self.data = []
            self.input_geo_maps = []
            self.input_score_maps = []
            self.input_training_masks = []
            video_cnt = [456, 450, 480, 780, 432, 408, 662, 504, 960, 480, 780, 301]
            video_all = ['Video_37_2_3', 'Video_51_7_4', 'Video_33_2_3', 'Video_47_6_4', 'Video_28_1_1', 'Video_40_2_3', 'Video_4_4_2', 'Video_42_2_3', 'Video_21_5_1', 'Video_54_7_4', 'Video_45_6_4', 'Video_16_3_2']
            self.video_name = video_all[video_start-1:video_end]
            self.cnt_frame = video_cnt[video_start-1:video_end]
            for i in range(video_start-1, video_end):
                if i == 0:
                    print("Loading and reshaping %d th groundtruth data "%(i))
                    self.input_geo_maps.append(ndimage.zoom(np.load(datapath + '/score.npy'), (1, 1/4, 1/4, 1)))
                    self.input_score_maps.append(ndimage.zoom(np.load(datapath + '/geo.npy'), (1, 1/4, 1/4)))
                    self.input_training_masks.append(ndimage.zoom(np.load(datapath + '/mask.npy'), (1, 1/4, 1/4)))
                else:
                    print("Loading and reshaping %d th groundtruth data "%(i))
                    self.input_geo_maps.append(ndimage.zoom(np.load(datapath + '/score%d.npy' % i), (1, 1/4, 1/4, 1)))
                    self.input_score_maps.append(ndimage.zoom(np.load(datapath + '/geo%d.npy' % i), (1, 1/4, 1/4)))
                    self.input_training_masks.append(ndimage.zoom(np.load(datapath + '/mask%d.npy' % i), (1, 1/4, 1/4)))


class HeatInputMul(object):
    """The input data."""
    def __init__(self, datapath, video_start, video_end, dimension):
        if dimension == 1:
            pass
        if dimension == 2:
            video_cnt = [456, 450, 480, 780, 432, 408, 662, 504, 960, 480, 780, 301]
            video_all = ['Video_37_2_3', 'Video_51_7_4', 'Video_33_2_3', 'Video_47_6_4', 'Video_28_1_1', 'Video_40_2_3', 'Video_4_4_2', 'Video_42_2_3', 'Video_21_5_1', 'Video_54_7_4', 'Video_45_6_4', 'Video_16_3_2']
            self.video_name = video_all[video_start-1:video_end]
            self.cnt_frame = video_cnt[video_start-1:video_end]
        if dimension == 0:
            video_cnt = [456, 450, 480, 780, 432, 408, 662, 504, 960, 480, 780, 301]
            video_all = ['Video_37_2_3', 'Video_51_7_4', 'Video_33_2_3', 'Video_47_6_4', 'Video_28_1_1', 'Video_40_2_3', 'Video_4_4_2', 'Video_42_2_3', 'Video_21_5_1', 'Video_54_7_4', 'Video_45_6_4', 'Video_16_3_2']
            self.video_name = video_all[video_start-1:video_end]
            self.cnt_frame = video_cnt[video_start-1:video_end]


class FrameInputMul(object):
    """ get input geo maps"""
    def __init__(self, datapath, video_start, video_end, from_source=True):
        if from_source is True:
            self.geo_maps, self.score_maps, self.training_masks, self.frame_info, self.video_name = reader.frame_producer_mul(datapath, video_start, video_end)
            print('All data loaded')
        else:
            None

"""Create the input data pipeline using `tf.data`"""


def _parse_function(filename, label, size):
    """Obtain the image from the filename (for both training and validation).
    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image_decoded = tf.image.decode_jpeg(image_string,u channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size])

    return resized_image, label


def train_preprocess(image, label, use_random_flip):
    """Image preprocessing for training.
    Apply the following operations:
        - Horizontally flip the image with probability 1/2
        - Apply random brightness and saturation
    """
    if use_random_flip:
        image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def input_fn(is_training, filenames, labels, params):
    """Input function for the SIGNS dataset.
    The filenames have format "{label}_IMG_{id}.jpg".
    For instance: "data_dir/2_IMG_4584.jpg".
    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the images, as ["data_dir/{label}_IMG_{id}.jpg"...]
        labels: (list) corresponding list of labels
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    num_samples = len(filenames)
    assert len(filenames) == len(labels), "Filenames and labels should have same length"

    # Create a Dataset serving batches of images and labels
    # We don't repeat for multiple epochs because we always train and evaluate for one epoch
    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    train_fn = lambda f, l: train_preprocess(f, l, params.use_random_flip)

    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .shuffle(num_samples)  # whole dataset into the buffer ensures good shuffling
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(filenames), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)  # make sure you always have one batch ready to serve
        )

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    return inputs
