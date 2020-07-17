import numpy as np

from mtcnn_base import TRTNet


def test_trt_net_gen_batches():
    trt_net = TRTNet(None, None, None, max_batch_size=4)

    image_arrs = [np.zeros((20, 20, 3)),] + [
        np.zeros((i, 20, 20, 3)) for i in range(1, 10)
    ]
    for image_arr in image_arrs:
        batches = list(trt_net._gen_batches(image_arr))
        expected_image_shape = (
            image_arr.shape[1:] if image_arr.ndim == 4 else image_arr.shape
        )
        expected_image_batch_size = image_arr.shape[0] if image_arr.ndim == 4 else 1
        assert all(b.shape[1:] == expected_image_shape for b in batches)
        assert sum(b.shape[0] for b in batches) == expected_image_batch_size


if __name__ == "__main__":
    test_trt_net_gen_batches()
