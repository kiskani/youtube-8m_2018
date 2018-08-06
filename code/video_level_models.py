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

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

flags.DEFINE_integer(
    "hidden_chain_layers", 4,
    "The number of layers used for HiddenChainModel")
flags.DEFINE_integer(
    "hidden_chain_relu_cells", 256,
    "The number of relu cells used for HiddenChainModel")

flags.DEFINE_integer(
    "deep_chain_layers", 3,
    "The number of layers used for DeepChainModel")
flags.DEFINE_integer(
    "deep_chain_relu_cells", 200,
    "The number of relu cells used for DeepChainModel")
flags.DEFINE_string(
    "deep_chain_relu_type", "relu",
    "The type of relu cells used for DeepChainModel (options are elu and relu)")

flags.DEFINE_bool(
    "deep_chain_use_length", False,
    "The number of relu cells used for DeepChainModel")

flags.DEFINE_integer(
    "divergence_model_count", 8,
    "The number of models used in divergence enhancement models")

class LogisticModel(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    output = slim.fully_connected(
        model_input, vocab_size, activation_fn=tf.nn.sigmoid,
        weights_regularizer=slim.l2_regularizer(l2_penalty))
    return {"predictions": output}

class MoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

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
    return {"predictions": final_probabilities}

class ChainMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
    num_supports = FLAGS.num_supports
    support_predictions = self.sub_model(model_input, num_supports, sub_scope=sub_scope+"-support")
    main_input = tf.concat([model_input, support_predictions], axis=1)
    main_predictions = self.sub_model(main_input, vocab_size, sub_scope=sub_scope+"-main")
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
        scope="gates"+sub_scope)
    expert_activations = slim.fully_connected(
        model_input,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts"+sub_scope)

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

class ChainMainReluMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
    num_supports = FLAGS.num_supports
    input_size = model_input.shape.as_list()[1]
    support_predictions = self.sub_model(model_input, num_supports, sub_scope=sub_scope+"-support")
    main_relu = slim.fully_connected(
        model_input,
        input_size,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="main-relu-"+sub_scope)
    main_input = tf.concat([main_relu, support_predictions], axis=1)
    main_predictions = self.sub_model(main_input, vocab_size, sub_scope=sub_scope+"-main")
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

class ChainSupportReluMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
    num_supports = FLAGS.num_supports
    input_size = model_input.shape.as_list()[1]
    support_relu = slim.fully_connected(
        model_input,
        input_size,
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="main-relu-"+sub_scope)
    support_predictions = self.sub_model(support_relu, num_supports, sub_scope=sub_scope+"-support")
    main_input = tf.concat([model_input, support_predictions], axis=1)
    main_predictions = self.sub_model(main_input, vocab_size, sub_scope=sub_scope+"-main")
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

class DeepChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None,
                   num_frames=None, **unused_params):
    num_supports = FLAGS.num_supports
    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    use_length = FLAGS.deep_chain_use_length

    if use_length:
      model_input = tf.concat([model_input, self.get_length_code(num_frames)], axis=1)

    next_input = model_input
    support_predictions = []
    for layer in range(num_layers):
      sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
      sub_relu = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)
      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      next_input = tf.concat([model_input, relu_norm], axis=1)
      support_predictions.append(sub_prediction)
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

class DeepCombineChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None,
                   dropout=False, keep_prob=None, noise_level=None,
                   num_frames=None,
                   **unused_params):

    num_supports = FLAGS.num_supports
    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    relu_type = FLAGS.deep_chain_relu_type
    use_length = FLAGS.deep_chain_use_length

    next_input = model_input
    support_predictions = []
    for layer in range(num_layers):
      sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer, dropout=dropout, keep_prob=keep_prob, noise_level=noise_level)
      sub_activation = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)

      if relu_type == "elu":
        sub_relu = tf.nn.elu(sub_activation)
      else: # default: relu
        sub_relu = tf.nn.relu(sub_activation)

      if noise_level is not None:
        print("adding noise to sub_relu, level = ", noise_level)
        sub_relu = sub_relu + tf.random_normal(tf.shape(sub_relu), mean=0.0, stddev=noise_level)

      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      next_input = tf.concat([next_input, relu_norm], axis=1)
      support_predictions.append(sub_prediction)
    main_predictions = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"-main")
    support_predictions = tf.concat(support_predictions, axis=1)
    return {"predictions": main_predictions, "support_predictions": support_predictions}

  def sub_model(self, model_input, vocab_size, num_mixtures=None,
                l2_penalty=1e-8, sub_scope="",
                dropout=False, keep_prob=None, noise_level=None,
                **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    if dropout:
      model_input = tf.nn.dropout(model_input, keep_prob=keep_prob)

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

class HiddenChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
    num_supports = FLAGS.num_supports
    num_layers = FLAGS.hidden_chain_layers
    relu_cells = FLAGS.hidden_chain_relu_cells

    next_input = model_input
    support_predictions = []
    for layer in range(num_layers):
      sub_relu = slim.fully_connected(
          next_input,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)
      sub_prediction = self.sub_model(sub_relu, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      next_input = tf.concat([model_input, relu_norm], axis=1)
      support_predictions.append(sub_prediction)
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


class HiddenCombineChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None, **unused_params):
    num_supports = FLAGS.num_supports
    num_layers = FLAGS.hidden_chain_layers
    relu_cells = FLAGS.hidden_chain_relu_cells

    next_input = model_input
    support_predictions = []
    for layer in range(num_layers):
      sub_relu = slim.fully_connected(
          next_input,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)
      sub_prediction = self.sub_model(sub_relu, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer)
      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      next_input = tf.concat([next_input, relu_norm], axis=1)
      support_predictions.append(sub_prediction)
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


class MlpMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   sub_scope="",
                   original_input=None,
                                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
    feature_dim = len(model_input.get_shape().as_list()) - 1
    num_features = model_input.get_shape().as_list()[feature_dim]
    layer1_size = 2048

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates"+sub_scope)
    expert_activations = slim.fully_connected(
        slim.fully_connected(
            model_input,
            num_features,
            activation_fn=tf.nn.relu,
            weights_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
            biases_initializer=tf.zeros_initializer(),
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="relu"+sub_scope),
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts"+sub_scope)

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
    return {"predictions": final_probabilities}

class DistillchainDeepCombineChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="", original_input=None,
                   dropout=False, keep_prob=None, noise_level=None,
                   distillation_predictions=None,
                   num_frames=None,
                   **unused_params):

    assert distillation_predictions is not None, "distillation feature must be used"

    num_supports = FLAGS.num_supports
    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    relu_type = FLAGS.deep_chain_relu_type
    use_length = FLAGS.deep_chain_use_length

    distill_relu = slim.fully_connected(
          distillation_predictions,
          relu_cells,
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"distillrelu")
    distill_norm = tf.nn.l2_normalize(distill_relu, dim=1)

    next_input = tf.concat([model_input, distill_norm], axis=1)
    support_predictions = []
    for layer in xrange(num_layers):
      sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer, dropout=dropout, keep_prob=keep_prob, noise_level=noise_level)
      sub_activation = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)

      if relu_type == "elu":
        sub_relu = tf.nn.elu(sub_activation)
      else: # default: relu
        sub_relu = tf.nn.relu(sub_activation)

      if noise_level is not None:
        print("adding noise to sub_relu, level = ", noise_level)
        sub_relu = sub_relu + tf.random_normal(tf.shape(sub_relu), mean=0.0, stddev=noise_level)

      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      next_input = tf.concat([next_input, relu_norm], axis=1)
      support_predictions.append(sub_prediction)
    main_predictions = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"-main")
    support_predictions = tf.concat(support_predictions, axis=1)
    return {"predictions": main_predictions, "support_predictions": support_predictions}

  def sub_model(self, model_input, vocab_size, num_mixtures=None,
                l2_penalty=1e-8, sub_scope="",
                dropout=False, keep_prob=None, noise_level=None,
                **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    if dropout:
      model_input = tf.nn.dropout(model_input, keep_prob=keep_prob)

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

class MultiTaskDivergenceMoeModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="ddcc", original_input=None,
                   dropout=False, keep_prob=None, noise_level=None,
                   num_frames=None, **unused_params):
    num_supports = FLAGS.num_supports
    num_models = FLAGS.divergence_model_count

    support_predictions = []
    for i in range(num_models):
      sub_prediction = self.sub_model(model_input,vocab_size, num_mixtures,
                                      l2_penalty, sub_scope+"%d"%i,
                                      dropout, keep_prob, noise_level)
      support_predictions.append(sub_prediction)
    support_predictions = tf.stack(support_predictions, axis=1)
    main_predictions = tf.reduce_mean(support_predictions, axis=1)
    return {"predictions": main_predictions, "support_predictions": support_predictions}

  def sub_model(self, model_input, vocab_size, num_mixtures=None,
                l2_penalty=1e-8, sub_scope="",
                dropout=False, keep_prob=None, noise_level=None,
                **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    if dropout:
      model_input = tf.nn.dropout(model_input, keep_prob=keep_prob)

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

class MultiTaskDivergenceDeepCombineChainModel(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self, model_input, vocab_size, num_mixtures=None,
                   l2_penalty=1e-8, sub_scope="ddcc", original_input=None,
                   dropout=False, keep_prob=None, noise_level=None,
                   num_frames=None, **unused_params):
    num_supports = FLAGS.num_supports
    num_models = FLAGS.divergence_model_count

    support_predictions = []
    for i in range(num_models):
      sub_prediction = self.sub_chain_model(model_input,vocab_size, num_mixtures,
                                      l2_penalty, sub_scope+"%d"%i, original_input,
                                      dropout, keep_prob, noise_level)
      support_predictions.append(sub_prediction)
    support_predictions = tf.stack(support_predictions, axis=1)
    main_predictions = tf.reduce_mean(support_predictions, axis=1)
    return {"predictions": main_predictions, "support_predictions": support_predictions}

  def sub_chain_model(self, model_input, vocab_size, num_mixtures=None,
                      l2_penalty=1e-8, sub_scope="", original_input=None, 
                      dropout=False, keep_prob=None, noise_level=None,
                      num_frames=None,
                      **unused_params):
    num_supports = FLAGS.num_supports
    num_layers = FLAGS.deep_chain_layers
    relu_cells = FLAGS.deep_chain_relu_cells
    relu_type = FLAGS.deep_chain_relu_type
    use_length = FLAGS.deep_chain_use_length

    input_list = [model_input]
    mean_activation = slim.fully_connected(
        model_input,
        relu_cells,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope=sub_scope+"relu-mean")
    mean_relu = tf.nn.relu(mean_activation)

    if noise_level is not None:
      print("adding noise to sub_relu, level = ", noise_level)
      mean_relu = mean_relu + tf.random_normal(tf.shape(mean_relu), mean=0.0, stddev=noise_level)

    mean_relu_norm = tf.nn.l2_normalize(mean_relu, dim=1)

    next_input = mean_relu_norm
    support_predictions = []
    for layer in range(num_layers):
      sub_prediction = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"prediction-%d"%layer, dropout=dropout, keep_prob=keep_prob, noise_level=noise_level)
      sub_activation = slim.fully_connected(
          sub_prediction,
          relu_cells,
          activation_fn=None,
          weights_regularizer=slim.l2_regularizer(l2_penalty),
          scope=sub_scope+"relu-%d"%layer)
      sub_relu = tf.nn.relu(sub_activation)

      if noise_level is not None:
        print("adding noise to sub_relu, level = ", noise_level)
        sub_relu = sub_relu + tf.random_normal(tf.shape(sub_relu), mean=0.0, stddev=noise_level)

      relu_norm = tf.nn.l2_normalize(sub_relu, dim=1)
      next_input = tf.concat([next_input, relu_norm], axis=1)
      support_predictions.append(sub_prediction)

    main_predictions = self.sub_model(next_input, vocab_size, sub_scope=sub_scope+"-main")
    support_predictions = tf.reduce_mean(tf.stack(support_predictions, axis=0), axis=0)

    ratio = min(FLAGS.support_loss_percent, 0.2)
    final_predictions = main_predictions * (1.0 - ratio) + support_predictions * ratio
    return final_predictions

  def sub_model(self, model_input, vocab_size, num_mixtures=None,
                l2_penalty=1e-8, sub_scope="",
                dropout=False, keep_prob=None, noise_level=None,
                **unused_params):
    num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

    if dropout:
      model_input = tf.nn.dropout(model_input, keep_prob=keep_prob)

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

