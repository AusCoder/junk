import logging
import functools
import operator
from collections import namedtuple, OrderedDict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import uff
import numpy as np
from tensorflow.keras import backend as K_tf, layers, Model
import tensorflow as tf
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

logger = logging.getLogger(__name__)


def clear_keras_session():
    K_tf.clear_session()


class FrozenGraphWithInfo:
    def __init__(
        self, name, frozen_graph, input_names, input_shapes, output_names, output_shapes
    ):
        self.name = name
        self.frozen_graph = frozen_graph
        self.input_names = input_names
        self.input_shapes = input_shapes
        self.output_names = output_names
        self.output_shapes = output_shapes

    def __repr__(self):
        rendered_attrs = ", ".join(
            f"{k}={v}" for k, v in self.__dict__.items() if k != "frozen_graph"
        )
        return f"{self.__class__.__name__}({rendered_attrs})"


def freeze_tf_keras_model(name: str, model) -> FrozenGraphWithInfo:
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
        name=name,
        frozen_graph=frozen_graph,
        input_names=input_names,
        input_shapes=input_shapes,
        output_names=output_names,
        output_shapes=output_shapes,
    )


def freeze_and_save_tf_keras_model(name, model, output_path: Path) -> None:
    graph = freeze_tf_keras_model(name, model)
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


def freeze_tf_keras_model_to_uff(name, model):
    frozen_graph_with_info = freeze_tf_keras_model(name, model)
    uff_graph_def = uff.from_tensorflow(frozen_graph_with_info.frozen_graph)
    return frozen_graph_with_info, uff_graph_def


def create_pnet(
    data_format: str, input_shape: Tuple[Optional[int], Optional[int]] = (None, None)
):
    if data_format == "channels_first":
        raise NotImplementedError("weight transpose might not work with channels_first")
        inpt = layers.Input(shape=(3, *input_shape), dtype="float32")
        prelu_shared_axes = [2, 3]
        softmax_axis = 1
        x = inpt
    elif data_format == "channels_last":
        inpt = layers.Input(shape=(*input_shape, 3), dtype="float32")
        prelu_shared_axes = [1, 2]
        softmax_axis = 3
        x = layers.Permute((2, 1, 3))(inpt)
    else:
        raise ValueError
    # conv1
    x = layers.Conv2D(
        filters=10, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(x)
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
    x = layers.Conv2D(
        filters=2, kernel_size=1, strides=1, padding="SAME", data_format=data_format
    )(prelu3_out)
    preprob = layers.Permute((2, 1, 3))(x)
    # UFF has issues with this Softmax. Try:
    # - flatten first
    # - try our own softmax fn
    # prob = layers.Softmax(axis=softmax_axis)(preprob)
    # conv4-2
    x = layers.Conv2D(
        filters=4, kernel_size=1, strides=1, padding="SAME", data_format=data_format
    )(prelu3_out)
    reg = layers.Permute((2, 1, 3))(x)
    # model
    return Model(inputs=inpt, outputs=[preprob, reg])


def create_rnet(data_format: str):
    if data_format == "channels_first":
        raise NotImplementedError("weight transpose might not work with channels_first")
        inpt = layers.Input(shape=(3, 24, 24), dtype="float32")
        prelu_shared_axes = [2, 3]
        x = inpt
    elif data_format == "channels_last":
        inpt = layers.Input(shape=(24, 24, 3), dtype="float32")
        prelu_shared_axes = [1, 2]
        x = layers.Permute((2, 1, 3))(inpt)
    else:
        raise ValueError
    # conv1
    x = layers.Conv2D(
        filters=28, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(x)
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

    # When transposing the weights, we need a layer like this:
    # x = layers.Permute((2, 1, 3))(x)

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
        raise NotImplementedError("weight transpose might not work with channels_first")
        inpt = layers.Input(shape=(3, 48, 48), dtype="float32")
        prelu_shared_axes = [2, 3]
        x = inpt
    elif data_format == "channels_last":
        inpt = layers.Input(shape=(48, 48, 3), dtype="float32")
        prelu_shared_axes = [1, 2]
        x = layers.Permute((2, 1, 3))(inpt)
    else:
        raise ValueError
    # conv1
    x = layers.Conv2D(
        filters=32, kernel_size=3, strides=1, padding="VALID", data_format=data_format
    )(x)
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

    # When transposing the weights, we need a layer like this:
    # x = layers.Permute((2, 1, 3))(x)

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
        ("PReLU1", prelue_weights_reshape(data_format)),
        ("conv2", None),
        ("PReLU2", prelue_weights_reshape(data_format)),
        ("conv3", None),
        ("PReLU3", prelue_weights_reshape(data_format)),
        ("conv4-1", None),
        ("conv4-2", None),
    ]
    return list(_gen_mtcnn_net_weights(weight_dict, layer_order))


def format_rnet_weights(weight_dict: Dict[str, Any], data_format: str):
    layer_order = [
        ("conv1", None),
        ("prelu1", prelue_weights_reshape(data_format)),
        ("conv2", None),
        ("prelu2", prelue_weights_reshape(data_format)),
        ("conv3", None),
        ("prelu3", prelue_weights_reshape(data_format)),
        ("conv4", None),
        ("prelu4", None),
        ("conv5-1", None),
        ("conv5-2", None),
    ]
    return list(_gen_mtcnn_net_weights(weight_dict, layer_order))


def format_onet_weights(weight_dict: Dict[str, Any], data_format: str):
    layer_order = [
        ("conv1", None),
        ("prelu1", prelue_weights_reshape(data_format)),
        ("conv2", None),
        ("prelu2", prelue_weights_reshape(data_format)),
        ("conv3", None),
        ("prelu3", prelue_weights_reshape(data_format)),
        ("conv4", None),
        ("prelu4", prelue_weights_reshape(data_format)),
        ("conv5", None),
        ("prelu5", None),
        ("conv6-1", None),
        ("conv6-2", None),
        ("conv6-3", None),
    ]
    return list(_gen_mtcnn_net_weights(weight_dict, layer_order))


def _gen_mtcnn_net_weights(
    weight_dict: Dict[str, Any],
    layers_with_reshape_fn: List[str],
    use_transpose: bool = False,
):
    """Formats original_facenet pnet weights to match the keras model
    """
    for layer_name, reshape_fn in layers_with_reshape_fn:
        if reshape_fn is None:
            reshape_fn = lambda x: x
        if layer_name.startswith("conv"):
            weights = weight_dict[layer_name]["weights"]

            if use_transpose and len(weights.shape) == 4:
                weights = np.transpose(weights, (1, 0, 2, 3))

            yield reshape_fn(weights)
            yield reshape_fn(weight_dict[layer_name]["biases"])
        elif layer_name.lower().startswith("prelu"):
            alpha = weight_dict[layer_name]["alpha"]
            yield reshape_fn(alpha)
        else:
            raise ValueError


def prelue_weights_reshape(data_format: str):
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
    """Base class for Keras Mtcnn networks
    """

    net_name = None
    _factories = {"pnet": create_pnet, "rnet": create_rnet, "onet": create_onet}
    _weight_formatters = {
        "pnet": format_pnet_weights,
        "rnet": format_rnet_weights,
        "onet": format_onet_weights,
    }

    def __init__(
        self,
        *,
        model_path: str,
        data_format: str = "channels_last",
        debug_input_output_dir=None,
        **kwargs,
    ):
        self.data_format = data_format
        self.debug_input_output_dir = debug_input_output_dir
        self.model = self._factories[self.net_name](data_format=data_format, **kwargs)
        weights_dict = np.load(model_path, allow_pickle=True, encoding="latin1").item()
        weights = self._weight_formatters[self.net_name](weights_dict, data_format)
        self.model.set_weights(weights)

    @property
    def inputs(self):
        return self.model.inputs

    @property
    def input_shapes(self):
        return tuple(
            tuple(int(x) for x in input_tensor.shape[1:])
            for input_tensor in self.inputs
        )

    @property
    def normalized_input_names(self):
        return tuple(
            input_tensor.name.replace(":0", "") for input_tensor in self.inputs
        )

    @property
    def outputs(self):
        return self.model.outputs

    @property
    def normalized_output_names(self):
        return tuple(t.name.replace(":0", "") for t in self.outputs)

    @classmethod
    def default_model(cls, **kwargs):
        model_path = str(_model_paths[cls.net_name])
        return cls(model_path=model_path, **kwargs)

    def predict(self, image_arr: np.ndarray) -> Any:
        # This transpose (NHWC -> NWHC) is required by the original caffe weights... this is pain
        # image_arr = np.transpose(image_arr, (0, 2, 1, 3))
        if self.data_format == "channels_first":
            assert False
            image_arr = image_arr.transpose((0, 3, 1, 2))
        print(f"first values for image: {image_arr.reshape(-1)[:10]}")
        out = self.model.predict(image_arr)
        self._save_input_output_if_debug(image_arr, out)
        if self.data_format == "channels_first":
            assert False
            out = [x.transpose((0, 2, 3, 1)) if x.ndim == 4 else x for x in out]
        return out

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def freeze(self) -> Any:
        return freeze_tf_keras_model(self.net_name, self.model)

    def freeze_and_save(self, output_path: Path) -> None:
        freeze_and_save_tf_keras_model(self.net_name, self.model, output_path)

    def freeze_to_uff(self):
        return freeze_tf_keras_model_to_uff(self.net_name, self.model)

    def _save_input_output_if_debug(self, image_arr, out):
        if self.debug_input_output_dir:
            self.debug_input_output_dir.mkdir(parents=True, exist_ok=True)
            self._save(image_arr, "input")

            try:
                prob, reg = out
            except ValueError:
                prob, reg, landmarks = out

            self._save(prob, "output-prob")
            self._save(reg, "output-reg")
            try:
                self._save(landmarks, "output-landmarks")
            except NameError:
                pass

    def _save(self, arr, desc):
        rendered_shape = "-".join(f"{x:d}" for x in arr.shape)
        input_path = self.debug_input_output_dir.joinpath(
            f"{self.net_name}_{rendered_shape}_{desc}.npy"
        )
        np.save(input_path, arr)


class KerasPNet(_KerasMTCNNNet):
    net_name = "pnet"


class KerasRNet(_KerasMTCNNNet):
    net_name = "rnet"


class KerasONet(_KerasMTCNNNet):
    net_name = "onet"


class TRTNet:
    """Base class for running Tensorrt nets
    """

    LOGGER = trt.Logger(trt.Logger.INFO)

    PREDEFINED_NET_SHAPES = {
        "conv_216x384": {
            "inputs": [("input_1", (384, 216, 3), trt.UffInputOrder.NHWC)],
            "outputs": ["conv2d/BiasAdd"],
        },
        "debug_net": {
            "inputs": [("input_1", (384, 216, 3), trt.UffInputOrder.NHWC)],
            "outputs": ["softmax/Softmax", "conv2d_4/BiasAdd"],
        }
        # "pnet_216x384": {
        #     "inputs": [("images", (160, 160, 3), trt.UffInputOrder.NHWC)],
        #     "outputs": ["preembeddings"],
        # }
    }

    def __init__(
        self,
        engine: trt.ICudaEngine,
        input_names: List[str],
        output_names: List[str],
        batch_size: int = 1,
    ):
        self.input_names = input_names
        self.output_names = output_names
        # TODO: provide a way to limit to max_batch_size from the cuda engine
        self.batch_size = batch_size

        self.engine = engine
        self.context = None
        self.stream = None
        self.h_inputs = None
        self.h_outputs = None
        self.d_inputs = None
        self.d_outputs = None
        self._names_with_idxs = None

    @staticmethod
    def create_from_uff_file(
        model_path: str, inputs: List[Tuple[str, Any, Any]], output_names: List[str]
    ):
        if isinstance(model_path, Path):
            model_path = str(model_path)
        builder = trt.Builder(TRTNet.LOGGER)
        parser, network = TRTNet.build_parser_and_network(builder, inputs, output_names)
        parser.parse(model_path, network)
        engine = builder.build_cuda_engine(network)
        input_names = [n for n, _, _ in inputs]
        return TRTNet(engine=engine, input_names=input_names, output_names=output_names)

    @staticmethod
    def build_parser_and_network(
        builder, inputs: List[Tuple[str, Any, Any]], output_names: List[str]
    ):
        builder.max_batch_size = 16
        # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
        builder.max_workspace_size = 1 << 30

        network = builder.create_network()
        parser = trt.UffParser()
        for name, shape, order in inputs:
            if order is not None:
                parser.register_input(name=name, shape=shape, order=order)
            else:
                parser.register_input(name=name, shape=shape)
        for name in output_names:
            parser.register_output(name=name)
        return parser, network

    @staticmethod
    def create_from_predefined_net_shapes(model_path: str, net_name: str):
        net_shape = TRTNet.PREDEFINED_NET_SHAPES[net_name]
        return TRTNet.create_from_uff_file(
            model_path, net_shape["inputs"], net_shape["outputs"]
        )

    def start(self):
        self.context = self.engine.create_execution_context()

        self._names_with_idxs = self._get_tensor_names()
        self.h_inputs = self._allocate_host(self.input_names, self._names_with_idxs)
        self.h_outputs = self._allocate_host(self.output_names, self._names_with_idxs)
        self.d_inputs = self._allocate_device(self.h_inputs)
        self.d_outputs = self._allocate_device(self.h_outputs)

        # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.

        # out_volume = _prod(int(d) for d in self.engine.get_binding_shape(1))

        # self.h_input = cuda.pagelocked_empty(in_volume, dtype=np.float32)
        # self.h_output = cuda.pagelocked_empty(out_volume, dtype=np.float32)
        # Allocate device memory for inputs and outputs.
        # self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        # self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        # Create a stream in which to copy inputs/outputs and run inference.
        self.stream = cuda.Stream()
        return self

    def _get_tensor_names(self):
        cur_idx = 0
        names_with_idxs = OrderedDict()
        while True:
            # TODO: check for the number of binds here to avoid an error log
            name = self.engine.get_binding_name(cur_idx)
            if not name:
                break
            names_with_idxs[name] = cur_idx
            cur_idx += 1
        return names_with_idxs

    def _allocate_host(self, names, names_with_idxs):
        host_allocations = {}
        for name in names:
            idx = names_with_idxs[name]
            volume = functools.reduce(
                operator.mul, (int(d) for d in self.engine.get_binding_shape(idx))
            )
            volume *= self.batch_size
            host_allocations[name] = cuda.pagelocked_empty(volume, dtype=np.float32)
        return host_allocations

    def _allocate_device(self, host_allocations):
        return {k: cuda.mem_alloc(v.nbytes) for k, v in host_allocations.items()}

    def predict(self, image_arr: np.ndarray):
        """Simplest predict that expects inputs to have length 1.

        Also performs no validation of input shape.
        """
        (inpt,) = self.h_inputs.values()
        inpt[...] = image_arr.reshape(-1)
        return self._run()

    def _run(self):
        # Transfer input data to the GPU.
        for name in self.h_inputs:
            cuda.memcpy_htod_async(
                self.d_inputs[name], self.h_inputs[name], self.stream
            )
        # Run inference.
        # bindings = [self.d_inputs[n] for n in self.input_names] + [
        #     self.d_outputs[n] for n in self.output_names
        # ]
        bindings = []
        for name in self._names_with_idxs:
            if name in self.d_inputs:
                h = self.d_inputs[name]
            else:
                h = self.d_outputs[name]
            bindings.append(int(h))
        # bindings = [int(x) for x in bindings]
        self.context.execute_async(
            bindings=bindings,
            batch_size=self.batch_size,
            stream_handle=self.stream.handle,
        )
        # Transfer predictions back from the GPU.
        for name in self.h_outputs:
            cuda.memcpy_dtoh_async(
                self.h_outputs[name], self.d_outputs[name], self.stream
            )
        # Synchronize the stream
        self.stream.synchronize()
        outputs = [self.h_outputs[n] for n in self.output_names]
        return outputs


if __name__ == "__main__":
    # Saves frozen graph of models
    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    outdir = Path.cwd().joinpath("data", "frzn")
    outdir.mkdir(exist_ok=True, parents=True)

    pnet216x384 = KerasPNet.default_model(input_shape=(216, 384))
    frozen_graph_with_info, uff_graph_def = pnet216x384.freeze_to_uff()
    print(frozen_graph_with_info)
    # pnet216x384.freeze_and_save(outdir.joinpath(f"pnet{216}x{384}.pb"))

    uff_outdir = Path.cwd().joinpath("data", "uff")
    uff_outdir.mkdir(exist_ok=True, parents=True)
    uff_outdir.joinpath("pnet_216x384.uff").write_bytes(uff_graph_def)
