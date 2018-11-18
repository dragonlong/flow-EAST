# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Grappler autoparallel optimizer."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import logging

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

FLAGS = tf.flags.FLAGS


def export_state_tuples(state_tuples, name):
    for state_tuple in state_tuples:
        tf.add_to_collection(name, state_tuple.c)
        tf.add_to_collection(name, state_tuple.h)


def import_state_tuples(state_tuples, name, num_replicas):
    restored = []
    for i in range(len(state_tuples) * num_replicas):
        c = tf.get_collection_ref(name)[2 * i + 0]
        h = tf.get_collection_ref(name)[2 * i + 1]
        restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
    return tuple(restored)


def with_prefix(prefix, name):
    """Adds prefix to name."""
    return "/".join((prefix, name))


def with_autoparallel_prefix(replica_id, name):
    return with_prefix("AutoParallel-Replica-%d" % replica_id, name)


class UpdateCollection(object):
    """Update collection info in MetaGraphDef for AutoParallel optimizer."""

    def __init__(self, metagraph, model):
        self._metagraph = metagraph
        self.replicate_states(model.initial_state_name)
        self.replicate_states(model.final_state_name)
        self.update_snapshot_name("variables")
        self.update_snapshot_name("trainable_variables")

    def update_snapshot_name(self, var_coll_name):
        var_list = self._metagraph.collection_def[var_coll_name]
        for i, value in enumerate(var_list.bytes_list.value):
            var_def = variable_pb2.VariableDef()
            var_def.ParseFromString(value)
            # Somehow node Model/global_step/read doesn't have any fanout and seems to
            # be only used for snapshot; this is different from all other variables.
            if var_def.snapshot_name != "Model/global_step/read:0":
                var_def.snapshot_name = with_autoparallel_prefix(
                        0, var_def.snapshot_name)
            value = var_def.SerializeToString()
            var_list.bytes_list.value[i] = value

    def replicate_states(self, state_coll_name):
        state_list = self._metagraph.collection_def[state_coll_name]
        num_states = len(state_list.node_list.value)
        for replica_id in range(1, FLAGS.num_gpus):
            for i in range(num_states):
                state_list.node_list.value.append(state_list.node_list.value[i])
        for replica_id in range(FLAGS.num_gpus):
            for i in range(num_states):
                index = replica_id * num_states + i
                state_list.node_list.value[index] = with_autoparallel_prefix(
                        replica_id, state_list.node_list.value[index])


def auto_parallel(metagraph, model):
    from tensorflow.python.grappler import tf_optimizer
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    rewriter_config.optimizers.append("autoparallel")
    rewriter_config.auto_parallel.enable = True
    rewriter_config.auto_parallel.num_replicas = FLAGS.num_gpus
    optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, metagraph)
    metagraph.graph_def.CopyFrom(optimized_graph)
    UpdateCollection(metagraph, model)


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
