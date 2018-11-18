import tensorflow as tf
###############################
# Config file
####################################################
global FLAGS
BASIC = "baisc"
CUDNN = "cudnn"
BLOCK = "block"
CUDNN_INPUT_LINEAR_MODE = "linear_input"
CUDNN_RNN_BIDIRECTION   = "bidirection"
CUDNN_RNN_UNIDIRECTION  = "unidirection"


def get_config(FLAGS):
    """Get model config."""
    config = None
    if FLAGS.model == "small":
        pass
    elif FLAGS.model == "medium":
        pass
    elif FLAGS.model == "large":
        pass
    elif FLAGS.model == "test":
        config = TestConfig()
    else:
        raise ValueError("Invalid model: %s", FLAGS.model)
    if FLAGS.rnn_mode:
        config.rnn_mode = FLAGS.rnn_mode
    if FLAGS.num_gpus != 1 or tf.__version__ < "1.3.0" :
        config.rnn_mode = BASIC
    return config


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 0.0001
    input_size    = 512
    max_grad_norm = 100
    num_layers    = 1
    num_steps     = 5
    output_size   = 360
    hidden_size   = 200
    epoch_size    = 100
    max_steps     = 100000
    keep_prob     = 1.0
    lr_decay      = 0.9999
    batch_size    = 8
    vocab_size    = 25600
    rnn_mode      = BLOCK
    shape         = [512, 512]
    filters       = 32
    kernel        = [3, 3]
    geometry      = 'RBOX'
