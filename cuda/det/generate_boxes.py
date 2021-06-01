"""Sample program doing just mtcnn box generation
"""

import numpy as np


def generate_bounding_box(prob, reg, threshold, scale):
    """Create boxes from net output

    Assume input is 1 image at a time.
    """
    stride = 2
    cell_size = 12

    # breakpoint()
    prob = prob[..., 1]
    mask = prob > threshold
    y, x = np.where(mask)
    indices = np.stack([x, y], axis=1)
    bbox = np.concatenate(
        ((indices * stride + 1) / scale, (indices * stride + cell_size) / scale), axis=1
    )
    prob = prob[mask]
    reg = reg[mask]
    return prob, reg, bbox


if __name__ == "__main__":
    # prob = np.array([[[0.1, 0.9], [0.8, 0.2]], [[0.7, 0.3], [0.3, 0.7]],])
    # reg = np.ones((2, 2, 2))
    # threshold = 0.5
    # scale = 1.0

    array_path = "./mtcnn-output-arrays/stage-one/prob-0.npy"
    (prob,) = np.load(array_path)
    reg = np.ones_like(prob)
    threshold = 0.95
    scale = 0.709

    _, _, bbox = generate_bounding_box(prob, reg, threshold, scale)
    print(f"bbox shape: {bbox.shape}")
    # print(bbox)
