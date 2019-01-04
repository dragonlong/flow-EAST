"""
Configurations for PWC optical flow
Copyright Xiaolong Li
Based on
https://github.com/philferriere/tfoptflow/blob/master/tfoptflow/pwcnet_eval_sm-6-2-multisteps-chairsthingsmix_mpisintelfinal.ipynb
written by Phil Ferriere, Copyright Phil Ferriere
"""


class cfg_sm(object):
    def __init__(self):
        self.mode = 'val_notrain'            # We're doing evaluation using the entire dataset for evaluation
        self.num_samples = 10                # Number of samples for error analysis
        self.sample_range = 10
        self.num_steps = 5
        self.batch_size = 8
        self.batch_size_per_gpu = 4
        self.input_size = 512
        self.ckpt_path = '/work/cascades/lxiaol9/ARC/PWC/checkpoints/pwcnet-sm-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-592000' # Model to eval
        self.gpu_devices = ['/device:GPU:0', '/device:GPU:1'] # We're doing the evaluation on a single GPU
        self.controller = '/device:GPU:0'

class cfg_lg(object):
    def __init__(self):
        self.mode = 'val_notrain'            # We're doing evaluation using the entire dataset for evaluation
        self.num_samples = 10                # Number of samples for error analysis
        self.sample_range = 10
        self.num_steps = 5
        self.batch_size = 16
        self.batch_size_per_gpu = 8
        self.input_size = 512
        self.ckpt_path = '/work/cascades/lxiaol9/ARC/PWC/checkpoints/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000' # Model to eval
        self.gpu_devices = ['/device:GPU:0', '/device:GPU:1'] # We're doing the evaluation on a single GPU
        self.controller = '/device:GPU:0'
