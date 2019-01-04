import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import _init_paths
import locality_aware_nms as nms_locality
import lanms
from utils.nms_highlevel import intersection
from utils import icdar


from utils.icdar import restore_rectangle


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def detect(score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]
    print('{} text boxes before nms, {} text boxes after nms'.format(text_box_restored.shape[0], boxes.shape[0]))

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


# used for iou searching and selecting TP, FP, FN #
def eval_single(bb_gt, bb_pred):
    """
    input params:
        bb_gt, numpy array, (batch_size, time_steps, 40*9)
        bb_pred, ((batch_size, time_steps, 40*9))
    """
    length_vect = bb_gt.shape[2]
    unit = 9
    num  = int(length_vect/unit)
    is_best = 0
    TP   = 0
    FP   = 0
    FN   = 0
    for i in range(num):
        for j in range(num):
            # pick out the best match
            iou = intersection(bb_gt[:,:,i*unit:i*unit+8], bb_pred[:,:,i*unit:i*unit+8])
            if iou>is_best:
                is_best = iou
        if iou > 0.5:
            TP = TP+1
        elif iou > 0:
            FP = FP+1
        else:
            FN = FN+1
    precision = TP/(TP+FP)
    recall    = TP/(TP+FN)
    F_measure = 2*precision*recall/(precision+recall)
    return precision, recall, F_measure


def eval_group(bb_gt, bb_pred):
    """

    """
    [m, n, _] = bb_gt.shape
    prec_c    = np.zeros((m, n), dtype=np.float32)
    recall_c  = np.zeros((m, n), dtype=np.float32)
    fmea_c    = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            p, r, f = eval_single(bb_gt[i, j, :], bb_pred[i, j, :])
            prec_c[i, j]   = p
            recall_c[i, j] = r
            fmea_c[i, j]   = f
    return prec_c, recall_c, fmea_c


def convert_vect(line):
    text_lines = []
    object_num = 40
    for n in range(0, object_num):
        tl = collections.OrderedDict()
        tl['x0'] = line[n * 9]
        tl['y0'] = line[n * 9 + 1]
        tl['x1'] = line[n * 9 + 2]
        tl['y1'] = line[n * 9 + 3]
        tl['x2'] = line[n * 9 + 4]
        tl['y2'] = line[n * 9 + 5]
        tl['x3'] = line[n * 9 + 6]
        tl['y3'] = line[n * 9 + 7]
        text_lines.append(tl)
    dict_coded = {'text_lines':text_lines}
    return dict_coded


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))
    return illu


def draw_illu_gt(illu, rst, p, r, f):
    if 'text_lines' in rst:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3
        fontColor = (255, 255, 255)
        lineType = 1
        infos = 'Precision ' + str(p)+ ', recall ' + str(r) + ', F_measure ' + str(f)
        cv2.putText(illu, infos,
                    (2, 20),
                    font,
                    0.5,
                    (255, 0, 0),
                    lineType)
        for t in rst['text_lines']:
            d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                          t['y2'], t['x3'], t['y3']], dtype='int32')
            d = d.reshape(-1, 2)
            cv2.polylines(illu, [d], isClosed=True, thickness=2, color=(0, 0, 0))
            bottomLeftCornerOfText = (int(t['x0']), int(t['y0']))

    return illu


# used for iou searching and selecting TP, FP, FN #
def eval_single_frame(target, box):
    """
    input params:
        target, python ordered dict
        box, sorted boxes dict from predictions
    """
    TP   = 0
    FP   = 0
    FN   = 0
    precision = 0
    recall = 0
    F_measure = 0
    if not len(target['text_lines'])==0:
        if not len(box['text_lines']) == 0:
            for t in target['text_lines']:
                d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                              t['y2'], t['x3'], t['y3']], dtype='int32')
                is_best = 0
                for m in box['text_lines']:
                    n = np.array([m['x0'], m['y0'], m['x1'], m['y1'], m['x2'],
                                  m['y2'], m['x3'], m['y3']], dtype='int32')

                    # pick out the best match
                    iou = intersection(n, d)
                    if iou>is_best:
                        is_best = iou
                if is_best > 0.5:
                    TP = TP+1
                elif is_best == 0:
                    FN = FN +1
                else:
                    FP = FP+1
            if TP > 0:
                precision = TP/(TP+FP)
                recall    = TP/(TP+FN)
                F_measure = 2*precision*recall/(precision+recall)
    return precision, recall, F_measure


# eval batch by batch
def model_eval(targets, predicts, video_name, video_index, frame_set):
    """
    :param targets:   [video, frames, vect(flexible)]
    :param predicts:  [video, frames, vect(9*40)]
    :param video_name:
    :param video_index:
    :param frame_set:
    :return:
    """
    global_path = '/media/dragonx/DataStorage/temporary/Video_text/ICDAR/train/'
    save_path = '/media/dragonx/DataStorage/ARC/EASTRNN/data/LSTM_train/'
    sample = video_name[video_index]
    precision = 0
    recall = 0
    f_measure = 0
    cnt_frame =len(frame_set)
    # going over all the frames, j is the starting frame index
    for i, j in enumerate(frame_set):
        # vector to ordered dict for evaluation
        target = convert_vect(targets[0, 0, :])
        predict = convert_vect(predicts[0, 0, :])
        p, r, f1 = eval_single_frame(target, predict)

        precision = precision + p
        recall = recall + r
        f_measure = f_measure + f1

        #################for debug use################
        video_path = global_path+sample+'.mp4'
        cap = cv2.VideoCapture(video_path)
        cap.set(2, j)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter(save_path + sample + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,\
                              (frame_width, frame_height))
        ret, frame = cap.read()
        # visualization


    return precision/cnt_frame, recall/cnt_frame, f_measure/cnt_frame


def main(argv=None):
    import os
    tf.app.flags.DEFINE_string('test_data_path', '/tmp/ch4_test_images/images/', '')
    tf.app.flags.DEFINE_string('gpu_list', '0', '')
    tf.app.flags.DEFINE_string('checkpoint_path', '/tmp/east_icdar2015_resnet_v1_50_rbox/', '')
    tf.app.flags.DEFINE_string('output_dir', '/tmp/ch4_test_images/images/', '')
    tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')
    FLAGS = tf.app.flags.FLAGS
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start

                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time
                print('[timing] {}'.format(duration))

                # save to file
                if boxes is not None:
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        '{}.txt'.format(
                            os.path.basename(im_fn).split('.')[0]))

                    with open(res_file, 'w') as f:
                        for box in boxes:
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1],
                            ))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])


if __name__ == '__main__':
    tf.app.run()
