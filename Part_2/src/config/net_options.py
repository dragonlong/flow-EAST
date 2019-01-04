"""
Configurations for model training
Copyright Xiaolong Li
"""

class sys_cfg(object):
    def __init__(self):
        self.mode = 'val_notrain'
        self.input_size = 512
        self.sample_range = 10
        self.num_steps = 5
