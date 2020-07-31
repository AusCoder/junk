import logging

from typing import Any, Optional
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def save_input_output(
    debug_input_output_dir: Path,
    *,
    inputs: Any,
    input_prefixes: Any,
    input_postfixes: Any = None,
    outputs: Any,
    output_prefixes: Any,
    output_postfixes: Any = None,
) -> None:
    if debug_input_output_dir:
        debug_input_output_dir.mkdir(parents=True, exist_ok=True)

        assert len(inputs) == len(input_prefixes)
        if input_postfixes:
            assert len(inputs) == len(input_postfixes)
        else:
            input_postfixes = [None for _ in inputs]

        assert len(outputs) == len(output_prefixes)
        if output_postfixes:
            assert len(outputs) == len(output_prefixes)
        else:
            output_postfixes = [None for _ in outputs]

        for input_, input_prefix, input_postfix in zip(
            inputs, input_prefixes, input_postfixes
        ):
            save_array(
                debug_input_output_dir,
                array=input_,
                prefix=input_prefix,
                postfix=input_postfix,
            )
        for output, output_prefix, output_postfix in zip(
            outputs, output_prefixes, output_postfixes
        ):
            save_array(
                debug_input_output_dir,
                array=output,
                prefix=output_prefix,
                postfix=output_postfix,
            )


def save_array(
    debug_input_output_dir: Path,
    *,
    array: Any,
    prefix: str,
    postfix: Optional[str] = None,
) -> None:
    debug_input_output_dir.mkdir(exist_ok=True, parents=True)
    rendered_shape = "-".join(f"{x:d}" for x in array.shape)
    post = ""
    if postfix:
        post = f"_{postfix}"
    path = debug_input_output_dir.joinpath(f"{prefix}_{rendered_shape}{post}.npy")
    logger.info("Saving array of shape %s to %s", path)
    np.save(path, array)
