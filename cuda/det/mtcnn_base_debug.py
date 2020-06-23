"""
Converting the mtcnn networks to TRT is not working.
These are some very simple networks to make sure they work.
"""

import logging
from collections import namedtuple
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tensorflow.keras import backend as K_tf, layers, Model

from mtcnn_base import (
    freeze_tf_keras_model,
    freeze_and_save_tf_keras_model,
    freeze_tf_keras_model_to_uff,
    prelue_weights_reshape,
    create_pnet,
    format_pnet_weights,
)


def create_debug_net(
    data_format: str, input_shape: Tuple[Optional[int], Optional[int]]
):
    if data_format == "channels_first":
        inpt = layers.Input(shape=(3, *input_shape), dtype="float32")
        prelu_shared_axes = [2, 3]
        softmax_axis = 1
    elif data_format == "channels_last":
        inpt = layers.Input(shape=(*input_shape, 3), dtype="float32")
        prelu_shared_axes = [1, 2]
        softmax_axis = 3
    else:
        raise ValueError
    # conv1
    x = layers.Conv2D(
        filters=10, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(inpt)
    x = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    x = layers.MaxPooling2D(
        pool_size=2, strides=2, padding="SAME", data_format=data_format
    )(x)
    # conv2
    x = layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(x)
    x = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    # conv3
    x = layers.Conv2D(
        filters=32, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(x)
    prelu3_out = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    # conv4-1
    preprob = layers.Conv2D(
        filters=2, kernel_size=1, strides=1, padding="SAME", data_format=data_format
    )(prelu3_out)
    # UFF has issues with this Softmax. Try:
    # - flatten first
    # - try our own softmax fn
    # prob = layers.Softmax(axis=softmax_axis)(preprob)
    # conv4-2
    reg = layers.Conv2D(
        filters=4, kernel_size=1, strides=1, padding="SAME", data_format=data_format
    )(prelu3_out)
    # model
    return Model(inputs=inpt, outputs=[preprob, reg])


def _create_debug_net(data_format: str):
    """This is rnet where we had trouble with the transposed weights.
    I'm going to check that pnet is working correctly first.
    I think the problem here is with the dense layers on the flattened layers.
    """
    if data_format == "channels_first":
        inpt = layers.Input(shape=(3, 24, 24), dtype="float32")
        prelu_shared_axes = [2, 3]
    elif data_format == "channels_last":
        inpt = layers.Input(shape=(24, 24, 3), dtype="float32")
        prelu_shared_axes = [1, 2]
    else:
        raise ValueError
    # conv1
    x = layers.Conv2D(
        filters=28, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(inpt)
    x = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="SAME", data_format=data_format
    )(x)
    # conv2
    x = layers.Conv2D(
        filters=48, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(x)
    x = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="VALID", data_format=data_format
    )(x)
    # conv3
    x = layers.Conv2D(
        filters=64, kernel_size=2, strides=1, padding="VALID", data_format=data_format
    )(x)
    x = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    # NB: Problem with TRT with this Permute
    # x = layers.Permute((2, 1, 3))(x)
    # conv4
    x = layers.Flatten(data_format=data_format)(x)
    x = layers.Dense(units=128)(x)
    prelu4_out = layers.PReLU()(x)
    # conv5-1
    preprob = layers.Dense(units=2)(prelu4_out)
    # prob = layers.Softmax(axis=1)(preprob)
    # conv5-2
    reg = layers.Dense(units=4)(prelu4_out)
    return Model(inputs=inpt, outputs=[preprob, reg])


def load_weights(data_format, model):
    model_path = Path(__file__).parent.joinpath("data", "orig", "det1.npy")
    weights_dict = np.load(model_path, allow_pickle=True, encoding="latin1").item()
    assert len(weights_dict.keys()) == len(set(k.lower() for k in weights_dict))
    weights_dict = {k.lower(): v for k, v in weights_dict.items()}

    weights = [
        # np.transpose(weights_dict["conv1"]["weights"], (1, 0, 2, 3)),
        weights_dict["conv1"]["weights"],
        weights_dict["conv1"]["biases"],
        prelue_weights_reshape(data_format)(weights_dict["prelu1"]["alpha"]),
        # np.transpose(weights_dict["conv2"]["weights"], (1, 0, 2, 3)),
        weights_dict["conv2"]["weights"],
        weights_dict["conv2"]["biases"],
        prelue_weights_reshape(data_format)(weights_dict["prelu2"]["alpha"]),
        # np.transpose(weights_dict["conv3"]["weights"], (1, 0, 2, 3)),
        weights_dict["conv3"]["weights"],
        weights_dict["conv3"]["biases"],
        prelue_weights_reshape(data_format)(weights_dict["prelu3"]["alpha"]),
        weights_dict["conv4-1"]["weights"],
        weights_dict["conv4-1"]["biases"],
        weights_dict["conv4-2"]["weights"],
        weights_dict["conv4-2"]["biases"],
        # weights_dict["prelu4"]["alpha"],
        # weights_dict["conv5-1"]["weights"],
        # weights_dict["conv5-1"]["biases"],
        # weights_dict["conv5-2"]["weights"],
        # weights_dict["conv5-2"]["biases"],
    ]
    model.set_weights(weights)
