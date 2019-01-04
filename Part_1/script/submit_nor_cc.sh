#!/bin/bash
#PBS -l walltime=01:20:00:00
#PBS -l nodes=1:ppn=20:gpus=2
#PBS -W group_list=cascades
#PBS -q v100_normal_q
#PBS -A DeepText
#PBS -m bea
#PBS -M lxiaol9@vt.edu
echo "job starts"
# let ipnport=($UID-6025)%65274
# echo $ipnport >> ipnport.txt
# echo "starting jupyter notebook"
# let ipnport=($UID-6025)%65274
# echo $ipnport >> ipnport.txt
# jupyter notebook --ip=$HOSTNAME --port=5034 --no-browser > jupyter.server
# module load Anaconda/4.2.0
module purge
module load gcc cmake
module load cuda/9.0.176 cudnn/7.1
source activate dlubu36
module list
unset LANG
export LANG=en_GB.UTF-8
cd /home/lxiaol9/videoText2018/flow-EAST/
#python train_rnn_east_crop1.py 2>&1 | tee output1.log
#python train_lstm_east_2013_1.py 2>&1 | tee output1.log
python train_flow_based_video_object_detection_recurrent2.py 2>&1 | tee output_recurrent2.log
