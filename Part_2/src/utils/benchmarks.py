"""
test benchmarks on data loading
"""
import matplotlib.pyplot as plt
from tensorpack.dataflow import AugmentImageComponent, BatchData, MultiThreadMapData, PrefetchDataZMQ, MultiProcessMapDataZMQ, dataset, imgaug
from tensorpack.input_source import QueueInput, StagingInput
from dataflow import MyDataFlow, preprocess
ICDAR2013 = '/work/cascades/lxiaol9/ARC/EAST/data/ICDAR2013/'
data_dir = '/work/cascades/lxiaol9/ARC/EAST/data/pre-processed/'
video_dir = ICDAR2013+'/train/'

def array_show(datapoint, nrows=2, ncols=4, step=0):
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
    fig.savefig(ICDAR2013+'vis/{}'.format(step)+".png")


# Option 1
def data_orig(num_readers=16, batch_size = 16, is_vis=False, is_print=False):

    # we may not need to process
    t1_start = time()
    dr = data_raw(video_dir, data_dir, is_print=True)
    df = MyDataFlow(raw_data=dr, num_steps=5, is_training=True)
    # df = MultiProcessMapDataZMQ(df, 4, preprocess)
    df = BatchData(df, batch_size)
    df = PrefetchDataZMQ(df, num_readers)
    # df = PlasmaGetData(df)
    df.reset_state()
    t1_end = time()
    print("data loader preparation costs {} seconds".format(t1_end - t1_start))
    step = 0
    time_start = time()
    for datapoint in df:
        if is_print is True:
            for j in range(len(datapoint)):
                print(datapoint[j].shape)
        if is_vis is True:
            array_show(datapoint)
        step = step + 1
        print("now passed {} seconds".format(time() - t1_end))
    average_time = (time.time() - time_start)/10
    return average_time


if "__name__" == "__main__":
        time_set = []
        print("Orginal test begin")
        average_time = data_orig(num_readers=8)
        time_set.append(average_time)
        average_time = data_orig(num_readers=8)
        time_set.append(average_time)
        print(time_set)
