# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")

flags.DEFINE_integer(
    "attention_matrix_rank", 3,
    "The rank of weight matrix used for AttentionMatrixModel.")
flags.DEFINE_integer(
    "attention_relu_cells", 128,
    "The rank of weight matrix used for AttentionMatrixModel.")

class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]

    denominators = tf.reshape(
        tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    avg_pooled = tf.reduce_sum(model_input,
                               axis=[1]) / denominators

    output = slim.fully_connected(
        avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(1e-8))
    return {"predictions": output}

class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)

class CnnDeepCombineChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def cnn(self,
          model_input,
          l2_penalty=1e-8,
          num_filters = [1024, 1024, 1024],
          filter_sizes = [1,2,3],
          sub_scope="",
          **unused_params):
    max_frames = model_input.get_shape().as_list()[1]
    num_features = model_input.get_shape().as_list()[2]

    shift_inputs = []
    for i in range(max(filter_sizes)):
      if i == 0:
        shift_inputs.append(model_input)
      else:
        shift_inputs.append(tf.pad(model_input, paddings=[[0,0],[i,0],[0,0]])[:,:max_frames,:])

    cnn_outputs = []
    for nf, fs in zip(num_filters, filter_sizes):
      sub_input = tf.concat(shift_inputs[:fs], axis=2)
      sub_filter = tf.get_variable(sub_scope+"cnn-filter-len%d"%fs,
                       shape=[num_features*fs, nf], dtype=tf.float32,
                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                       regularizer=tf.contrib.layers.l2_regularizer(l2_penalty))
      cnn_outputs.append(tf.einsum("ijk,kl->ijl", sub_input, sub_filter))

    cnn_output = tf.concat(cnn_outputs, axis=2)
    return cnn_output

  def create_model(self, model_input, vocab_size, num_frames, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
    num_supports = FLAGS.num_supports
    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    max_frames = model_input.get_shape().as_list()[1]
    relu_layers = []
    support_predictions = []

    mask = self.get_mask(max_frames, num_frames)
    mean_input = tf.einsum("ijk,ij->ik", model_input, mask) \
                      / tf.expand_dims(tf.cast(num_frames, dtype=tf.float32), dim=1)
    mean_relu = slim.fully_connected(
          mean_input,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"mean-relu")
    mean_relu_norm = tf.nn.l2_normalize(mean_relu, dim=1)
    relu_layers.append(mean_relu_norm)

    cnn_output = self.cnn(model_input, num_filters=[relu_cells,relu_cells,relu_cells*2], filter_sizes=[1,2,3], sub_scope=sub_scope+"cnn0")
    max_cnn_output = tf.reduce_max(cnn_output, axis=1)
    normalized_cnn_output = tf.nn.l2_normalize(max_cnn_output, dim=1)
    next_input = normalized_cnn_output

    for layer in range(num_layers):
      sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
      support_predictions.append(sub_prediction)

      sub_relu = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)
      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      relu_layers.append(relu_norm)

      cnn_output = self.cnn(model_input, num_filters=[relu_cells,relu_cells,relu_cells*2], filter_sizes=[1,2,3], sub_scope=sub_scope+"cnn%d"%(layer+1))
      max_cnn_output = tf.reduce_max(cnn_output, axis=1)
      normalized_cnn_output = tf.nn.l2_normalize(max_cnn_output, dim=1)
      next_input = tf.concat([mean_input, normalized_cnn_output] + relu_layers, axis=1)

    main_predictions = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"-main")
    support_predictions = tf.concat(support_predictions, axis=1)
    return {"predictions": main_predictions, "support_predictions": support_predictions}

  def sub_model(self, model_input, vocab_size, num_mixtures=None,
                l2_penalty=1e-8, sub_scope="", **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates-"+sub_scope)
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts-"+sub_scope)

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return final_probabilities

  def get_mask(self, max_frames, num_frames):
    mask_array = []
    for i in range(max_frames + 1):
      tmp = [0.0] * max_frames
      for j in range(i):
        tmp[j] = 1.0
      mask_array.append(tmp)
    mask_array = np.array(mask_array)
    mask_init = tf.constant_initializer(mask_array)
    mask_emb = tf.get_variable("mask_emb", shape = [max_frames + 1, max_frames],
            dtype = tf.float32, trainable = False, initializer = mask_init)
    mask = tf.nn.embedding_lookup(mask_emb, num_frames)
    return mask

class AttentionMoeMatrixModel(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   sub_scope="",
                   original_input=None,
                   **unused_params):

    num_relu = FLAGS.attention_relu_cells
    num_methods = model_input.get_shape().as_list()[-1]
    num_features = model_input.get_shape().as_list()[-2]
    num_mixtures = FLAGS.moe_num_mixtures
    attention_matrix_rank = FLAGS.attention_matrix_rank

    # gating coefficients
    mean_input = tf.reduce_mean(model_input, axis=2)
    std_input = tf.reduce_sum(
        tf.square(model_input - tf.expand_dims(mean_input, dim=2)), 
        axis=2) / (num_methods - 1)

    # relu
    original_relu = self.relu(original_input, num_relu, sub_scope="origin")
    mean_relu = self.relu(mean_input, num_relu, sub_scope="mean")
    std_relu = self.relu(std_input, num_relu, sub_scope="std")

    # normalize
    original_relu = tf.nn.l2_normalize(original_relu)
    mean_relu = tf.nn.l2_normalize(mean_relu)
    std_relu = tf.nn.l2_normalize(std_relu)

    ## batch_size x moe_num_mixtures
    gate_activations = slim.fully_connected(
        tf.concat([original_relu, mean_relu, std_relu], axis=1),
        num_mixtures,
        activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates"+sub_scope)

    # matrix
    weight_x = tf.get_variable("ensemble_weightx", 
        shape=[num_mixtures, num_features, attention_matrix_rank],
        regularizer=slim.l2_regularizer(l2_penalty))
    weight_y = tf.get_variable("ensemble_weighty",
        shape=[num_mixtures, attention_matrix_rank, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    ## moe_num_mixtures x num_features x num_methods
    weight_xy = tf.einsum("ijl,ilk->ijk", weight_x, weight_y)

    # weight
    gated_weight_xy = tf.einsum("ij,jkl->ikl", gate_activations, weight_xy)
    weight = tf.nn.softmax(gated_weight_xy)

    # weighted output
    output = tf.reduce_sum(weight * model_input, axis=2)
    return {"predictions": output}

  def relu(self, model_input, relu_cells,
               l2_penalty=1e-8, sub_scope=""):
    sub_activation = slim.fully_connected(
        model_input,
        relu_cells,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="relu-"+sub_scope)
    return sub_activation

class AttentionMatrixModel(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   sub_scope="",
                   original_input=None,
                   **unused_params):

    num_methods = model_input.get_shape().as_list()[-1]
    num_features = model_input.get_shape().as_list()[-2]
    num_mixtures = FLAGS.moe_num_mixtures
    attention_matrix_rank = FLAGS.attention_matrix_rank

    # gating coefficients
    original_input = tf.nn.l2_normalize(original_input, dim=1)
    mean_output = tf.reduce_mean(model_input, axis=2)
    ## batch_size x moe_num_mixtures
    gate_activations = slim.fully_connected(
        tf.concat([original_input, mean_output], axis=1),
        num_mixtures,
        activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates"+sub_scope)

    # matrix
    weight_x = tf.get_variable("ensemble_weightx", 
        shape=[num_mixtures, num_features, attention_matrix_rank],
        regularizer=slim.l2_regularizer(l2_penalty))
    weight_y = tf.get_variable("ensemble_weighty", 
        shape=[num_mixtures, attention_matrix_rank, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))
    ## moe_num_mixtures x num_features x num_methods
    weight_xy = tf.einsum("ijl,ilk->ijk", weight_x, weight_y)

    # weight
    gated_weight_xy = tf.einsum("ij,jkl->ikl", gate_activations, weight_xy)
    weight = tf.nn.softmax(gated_weight_xy)
    
    # weighted output
    output = tf.reduce_sum(weight * model_input, axis=2)
    return {"predictions": output}

class AttentionRectifiedLinearModel(models.BaseModel):

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   sub_scope="",
                   original_input=None, 
                   **unused_params):

    num_methods = model_input.get_shape().as_list()[-1]
    num_features = model_input.get_shape().as_list()[-2]
    num_mixtures = FLAGS.moe_num_mixtures

    # gating coefficients
    original_input = tf.nn.l2_normalize(original_input, dim=1)
    mean_output = tf.reduce_mean(model_input, axis=2)
    ## batch_size x moe_num_mixtures
    gate_activations = slim.fully_connected(
        tf.concat([original_input, mean_output], axis=1),
        num_mixtures,
        activation_fn=tf.nn.softmax,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates"+sub_scope)

    # matrix
    weight_var = tf.get_variable("ensemble_weight",
        shape=[num_mixtures, num_methods],
        regularizer=slim.l2_regularizer(l2_penalty))

    # weight
    gated_weight = tf.einsum("ij,jk->ik", gate_activations, weight_var)
    rl_gated_weight = tf.nn.relu(gated_weight) + 1e-9
    sum_gated_weight = tf.reduce_sum(rl_gated_weight, axis=1, keep_dims=True)
    weight = rel_gated_weight / sum_gated_weight
    
    # weighted output
    output = tf.einsum("ik,ijk->ij", weight, model_input)
    return {"predictions": output}
