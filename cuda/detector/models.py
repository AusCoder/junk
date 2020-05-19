import logging
from collections import namedtuple
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from tensorflow.keras import backend as K_tf, layers, Model
import tensorflow as tf

logger = logging.getLogger(__name__)


FrozenGraphWithInfo = namedtuple(
    "FrozenGraphWithInfo",
    ["frozen_graph", "input_names", "input_shapes", "output_names", "output_shapes"],
)


def freeze_tf_keras_model(model) -> FrozenGraphWithInfo:
    """
    Note: there are potentially 2 versions of keras floating around
    one from Keras and one from tensorflow.keras.

    I don't know if these share sessions (probably not) so this method
    explicitly uses tensorflow.keras
    """
    sess = K_tf.get_session()
    graph_def = sess.graph.as_graph_def()
    input_names = [t.name.split(":")[0] for t in model.inputs]
    output_names = [t.name.split(":")[0] for t in model.outputs]
    input_shapes = [tuple(x.value for x in t.shape) for t in model.inputs]
    output_shapes = [tuple(x.value for x in t.shape) for t in model.outputs]
    frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, output_names
    )
    return FrozenGraphWithInfo(
        frozen_graph=frozen_graph,
        input_names=input_names,
        input_shapes=input_shapes,
        output_names=output_names,
        output_shapes=output_shapes,
    )


def freeze_and_save_tf_keras_model(model, output_path: Path) -> None:
    graph = freeze_tf_keras_model(model)
    tf.train.write_graph(
        graph.frozen_graph,
        str(output_path.parent),
        str(output_path.name),
        as_text=False,
    )

    logger.info(
        f"Froze tensorflow.keras graph. Input names: {graph.input_names}. Input shapes: {graph.input_shapes}"
    )
    logger.info(
        f"Output names: {graph.output_names}. Output shapes: {graph.output_shapes}."
    )
    logger.info(f"Output path: {output_path}")


def create_pnet(data_format: str, input_shape: Tuple[Optional[int], Optional[int]]):
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
    prob = layers.Softmax(axis=softmax_axis)(preprob)
    # conv4-2
    reg = layers.Conv2D(
        filters=4, kernel_size=1, strides=1, padding="SAME", data_format=data_format
    )(prelu3_out)
    # model
    return Model(inputs=inpt, outputs=[prob, reg])


def create_rnet(data_format: str):
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
    # conv4
    x = layers.Flatten(data_format=data_format)(x)
    x = layers.Dense(units=128)(x)
    prelu4_out = layers.PReLU()(x)
    # conv5-1
    preprob = layers.Dense(units=2)(prelu4_out)
    prob = layers.Softmax(axis=1)(preprob)
    # conv5-2
    reg = layers.Dense(units=4)(prelu4_out)
    return Model(inputs=inpt, outputs=[prob, reg])


def create_onet(data_format: str):
    if data_format == "channels_first":
        inpt = layers.Input(shape=(3, 48, 48), dtype="float32")
        prelu_shared_axes = [2, 3]
    elif data_format == "channels_last":
        inpt = layers.Input(shape=(48, 48, 3), dtype="float32")
        prelu_shared_axes = [1, 2]
    else:
        raise ValueError
    # conv1
    x = layers.Conv2D(
        filters=32, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(inpt)
    x = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="SAME", data_format=data_format
    )(x)
    # conv2
    x = layers.Conv2D(
        filters=64, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(x)
    x = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    x = layers.MaxPooling2D(
        pool_size=3, strides=2, padding="VALID", data_format=data_format
    )(x)
    # conv3
    x = layers.Conv2D(
        filters=64, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(x)
    x = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    x = layers.MaxPooling2D(
        pool_size=2, strides=2, padding="SAME", data_format=data_format
    )(x)
    # conv4
    x = layers.Conv2D(
        filters=128, kernel_size=2, strides=1, padding="VALID", data_format=data_format
    )(x)
    x = layers.PReLU(shared_axes=prelu_shared_axes)(x)
    # conv5
    x = layers.Flatten(data_format=data_format)(x)
    x = layers.Dense(units=256)(x)
    prelu5_out = layers.PReLU()(x)
    # conv6-1
    preprob = layers.Dense(units=2)(prelu5_out)
    prob = layers.Softmax(axis=1)(preprob)
    # conv6-2
    reg = layers.Dense(units=4)(prelu5_out)
    # conv6-3
    landmarks = layers.Dense(units=10)(prelu5_out)
    return Model(inputs=inpt, outputs=[prob, reg, landmarks])


def format_pnet_weights(weight_dict: Dict[str, Any], data_format: str):
    layer_order = [
        ("conv1", None),
        ("PReLU1", _conv_prelue_reshape(data_format)),
        ("conv2", None),
        ("PReLU2", _conv_prelue_reshape(data_format)),
        ("conv3", None),
        ("PReLU3", _conv_prelue_reshape(data_format)),
        ("conv4-1", None),
        ("conv4-2", None),
    ]
    return list(_gen_mtcnn_net_weights(weight_dict, layer_order))


def format_rnet_weights(weight_dict: Dict[str, Any], data_format: str):
    layer_order = [
        ("conv1", None),
        ("prelu1", _conv_prelue_reshape(data_format)),
        ("conv2", None),
        ("prelu2", _conv_prelue_reshape(data_format)),
        ("conv3", None),
        ("prelu3", _conv_prelue_reshape(data_format)),
        ("conv4", None),
        ("prelu4", None),
        ("conv5-1", None),
        ("conv5-2", None),
    ]
    return list(_gen_mtcnn_net_weights(weight_dict, layer_order))


def format_onet_weights(weight_dict: Dict[str, Any], data_format: str):
    layer_order = [
        ("conv1", None),
        ("prelu1", _conv_prelue_reshape(data_format)),
        ("conv2", None),
        ("prelu2", _conv_prelue_reshape(data_format)),
        ("conv3", None),
        ("prelu3", _conv_prelue_reshape(data_format)),
        ("conv4", None),
        ("prelu4", _conv_prelue_reshape(data_format)),
        ("conv5", None),
        ("prelu5", None),
        ("conv6-1", None),
        ("conv6-2", None),
        ("conv6-3", None),
    ]
    return list(_gen_mtcnn_net_weights(weight_dict, layer_order))


def _gen_mtcnn_net_weights(
    weight_dict: Dict[str, Any], layers_with_reshape_fn: List[str]
):
    """Formats original_facenet pnet weights to match the keras model
    """
    for layer_name, reshape_fn in layers_with_reshape_fn:
        if reshape_fn is None:
            reshape_fn = lambda x: x
        if layer_name.startswith("conv"):
            yield reshape_fn(weight_dict[layer_name]["weights"])
            yield reshape_fn(weight_dict[layer_name]["biases"])
        elif layer_name.lower().startswith("prelu"):
            alpha = weight_dict[layer_name]["alpha"]
            yield reshape_fn(alpha)
        else:
            raise ValueError


def _conv_prelue_reshape(data_format: str):
    if data_format == "channels_first":

        def reshape(x):
            return x.reshape((len(x), 1, 1))

    elif data_format == "channels_last":

        def reshape(x):
            return x.reshape((1, 1, len(x)))

    else:
        raise ValueError
    return reshape


_original_facenet_model_dir = Path(__file__).parent.joinpath("data", "orig")
_model_paths = {
    "simple_pnet": _original_facenet_model_dir.joinpath("det1.npy"),
    "pnet": _original_facenet_model_dir.joinpath("det1.npy"),
    "rnet": _original_facenet_model_dir.joinpath("det2.npy"),
    "onet": _original_facenet_model_dir.joinpath("det3.npy"),
}


class _KerasMTCNNNet:
    _factories = {"pnet": create_pnet, "rnet": create_rnet, "onet": create_onet}
    _weight_formatters = {
        "pnet": format_pnet_weights,
        "rnet": format_rnet_weights,
        "onet": format_onet_weights,
    }

    def __init__(
        self, model_path: str, *args, data_format: str = "channels_last", **kwargs
    ):
        self.data_format = data_format
        self.model = self._factories[self._net_name](
            *args, data_format=data_format, **kwargs
        )
        weights_dict = np.load(model_path, allow_pickle=True, encoding="latin1").item()
        weights = self._weight_formatters[self._net_name](weights_dict, data_format)
        self.model.set_weights(weights)

    @classmethod
    def default_model(cls, *args, **kwargs):
        model_path = _model_paths[cls._net_name]
        return cls(str(model_path), *args, **kwargs)

    def predict(self, image_arr: np.ndarray) -> Any:
        # This transpose (NHWC -> NWHC) is required by the original caffe weights... this is pain
        image_arr = np.transpose(image_arr, (0, 2, 1, 3))
        if self.data_format == "channels_first":
            image_arr = image_arr.transpose((0, 3, 1, 2))
        out = self.model.predict(image_arr)
        if self.data_format == "channels_first":
            out = [x.transpose((0, 2, 3, 1)) if x.ndim == 4 else x for x in out]
        return out

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def freeze(self) -> Any:
        return freeze_tf_keras_model(self.model)

    def freeze_and_save(self, output_path: Path) -> None:
        freeze_and_save_tf_keras_model(self.model, output_path)


class KerasPNet(_KerasMTCNNNet):
    _net_name = "pnet"

    def __init__(
        self,
        *args,
        input_shape: Tuple[Optional[int], Optional[int]] = (None, None),
        **kwargs,
    ) -> None:
        # We swap the channel order because the default weights require this
        h, w = input_shape
        super().__init__(*args, input_shape=(w, h), **kwargs)

    def predict(self, image_arr: np.ndarray):
        prob, reg = super().predict(image_arr)
        prob = np.transpose(prob, (0, 2, 1, 3))
        reg = np.transpose(reg, (0, 2, 1, 3))
        return prob, reg


class KerasRNet(_KerasMTCNNNet):
    _net_name = "rnet"


class KerasONet(_KerasMTCNNNet):
    _net_name = "onet"


if __name__ == "__main__":
    # Saves frozen graph of models
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    outdir = Path.cwd().joinpath("data", "frzn")
    outdir.mkdir(exist_ok=True, parents=True)

    pnet216x384 = KerasPNet.default_model(input_shape=(216, 384))
    pnet216x384.freeze_and_save(outdir.joinpath(f"pnet{216}x{384}.pb"))
