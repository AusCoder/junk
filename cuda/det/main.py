import logging
from pathlib import Path

import click
import cv2
import numpy as np

from mtcnn_base import KerasPNet, KerasRNet, KerasONet, clear_keras_session, TRTNet
from mtcnn_network import (
    compute_height_width_at_scale,
    compute_scales,
    DEFAULT_MIN_SIZE,
    DEFAULT_SCALE_FACTOR,
    MTCNN,
)

DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280


def _setup_logging():
    logging.basicConfig()
    for name in [__name__, "models", "network"]:
        logging.getLogger(name).setLevel(logging.INFO)


@click.group()
def main():
    pass


@main.command()
def print_scales():
    height, width = DEFAULT_HEIGHT, DEFAULT_WIDTH
    scales = compute_scales(DEFAULT_MIN_SIZE, DEFAULT_SCALE_FACTOR, height, width)
    click.echo(f"Scales for {height} x {width}")
    for scale in scales:
        scaled_height, scaled_width = compute_height_width_at_scale(
            scale, height, width
        )
        click.echo(
            f"Scale: {scale:.5f}. Scaled height width: "
            f"{scaled_height:d} x {scaled_width:d}"
        )


@main.command()
def save_uff():
    _setup_logging()

    tf_outdir = Path.cwd().joinpath("data", "frzn")
    tf_outdir.mkdir(exist_ok=True, parents=True)

    uff_outdir = Path.cwd().joinpath("data", "uff")
    uff_outdir.mkdir(exist_ok=True, parents=True)

    height, width = DEFAULT_HEIGHT, DEFAULT_WIDTH
    scales = compute_scales(DEFAULT_MIN_SIZE, DEFAULT_SCALE_FACTOR, height, width)

    frozen_graph_infos = []

    for scale in scales:
        scaled_height, scaled_width = compute_height_width_at_scale(
            scale, height, width
        )
        graph_name = f"pnet_{scaled_height}x{scaled_width}"

        pnet = KerasPNet.default_model(input_shape=(scaled_height, scaled_width))
        frozen_graph_info, uff_graph_def = pnet.freeze_to_uff()
        frozen_graph_infos.append(frozen_graph_info)

        uff_outdir.joinpath(f"{graph_name}.uff").write_bytes(uff_graph_def)
        pnet.freeze_and_save(tf_outdir.joinpath(f"pnet_{graph_name}.pb"))
        clear_keras_session()

    rnet = KerasRNet.default_model()
    frozen_graph_info, uff_graph_def = rnet.freeze_to_uff()
    uff_outdir.joinpath(f"rnet.uff").write_bytes(uff_graph_def)
    rnet.freeze_and_save(tf_outdir.joinpath(f"rnet.pb"))
    frozen_graph_infos.append(frozen_graph_info)

    onet = KerasONet.default_model()
    frozen_graph_info, uff_graph_def = onet.freeze_to_uff()
    uff_outdir.joinpath(f"onet.uff").write_bytes(uff_graph_def)
    onet.freeze_and_save(tf_outdir.joinpath(f"onet.pb"))
    frozen_graph_infos.append(frozen_graph_info)

    for info in frozen_graph_infos:
        click.echo(info)


@main.command()
@click.option(
    "-o", "--debug-input-output-dir", help="Directory to write debug net input output"
)
def run_keras(debug_input_output_dir):
    debug_input_output_dir = Path(debug_input_output_dir)
    DEFAULT_IMAGE_SIZE = (720, 1280)

    mtcnn = MTCNN.keras_predictors(
        debug_input_output_dir=debug_input_output_dir, image_size=DEFAULT_IMAGE_SIZE
    )
    image_path = Path.cwd().parent.parent.joinpath("tests", "data", "execs.jpg")
    assert image_path.exists()
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    result = mtcnn.predict(image)
    click.echo(result)


@main.command()
def debug_verify_trt():
    import tensorrt as trt

    from mtcnn_base_debug import (
        create_debug_net,
        load_weights,
        freeze_tf_keras_model_to_uff,
    )

    input_shape = (384, 216)
    data_format = "channels_last"
    keras_model = create_debug_net(data_format, input_shape)
    load_weights(data_format, keras_model)

    _, uff_graph_def = freeze_tf_keras_model_to_uff("debug_net", keras_model)
    output_dir = Path(__file__).parent.joinpath("data", "debug_uff")
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir.joinpath("debug_net.uff").write_bytes(uff_graph_def)

    uff_model_path = Path(__file__).parent.joinpath(
        "data", "debug_uff", "debug_net.uff"
    )
    assert uff_model_path.exists()
    trt_model = TRTNet.create_from_uff_file(
        str(uff_model_path),
        inputs=[("input_1", (*input_shape, 3), trt.UffInputOrder.NHWC)],
        output_names=[t.name.replace(":0", "") for t in keras_model.outputs],
    )

    trt_model.start()

    _verify_keras_and_trt_models(keras_model, trt_model)


def _verify_keras_and_trt_models(keras_model, trt_model):
    inpt = np.zeros((1, 384, 216, 3), dtype=np.float32)
    keras_outputs = keras_model.predict(inpt)
    if not isinstance(keras_outputs, list):
        keras_outputs = [keras_outputs]
    trt_outputs = trt_model.predict(inpt)
    assert len(keras_outputs) == len(trt_outputs)
    for keras_output, trt_output in zip(keras_outputs, trt_outputs):
        print(
            f"Output shape: {keras_output.shape}. First 10: {keras_output.reshape(-1)[:10]}"
        )
        np.testing.assert_allclose(
            keras_output.reshape(-1), trt_output.reshape(-1), atol=1e-6, rtol=1e-6
        )
    click.echo("Keras TRT output verified")


if __name__ == "__main__":
    main()
