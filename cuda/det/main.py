import logging
from pathlib import Path

import click
import cv2

from models import KerasPNet, KerasRNet, KerasONet, clear_keras_session
from network import (
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
@main.command(
    "-o", "--debug-input-output-dir", help="Directory to write debug net input output"
)
def run(debug_input_output_dir):
    mtcnn = MTCNN.keras_predictors(debug_input_output_dir=debug_input_output_dir)
    image_path = Path.cwd().parent.parent.joinpath("tests", "data", "execs.jpg")
    assert image_path.exists()
    image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    result = mtcnn.predict(image)
    click.echo(result)


if __name__ == "__main__":
    main()
