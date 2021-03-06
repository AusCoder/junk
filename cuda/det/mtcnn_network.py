"""Implements the main network logic required for mtcnn
"""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import scipy.special
import numpy as np

from mtcnn_utils import save_array, save_input_output


DEFAULT_MIN_SIZE = 40
DEFAULT_SCALE_FACTOR = 0.709


def _debug_print_and_exit(arr):
    print(arr.reshape(-1)[:10])
    import sys

    sys.exit(1)


class MTCNN:
    def __init__(
        self,
        pnet,
        rnet,
        onet,
        *,
        thresholds: Optional[List[float]] = None,
        factor: float = DEFAULT_SCALE_FACTOR,
        min_face_magnitude: int = DEFAULT_MIN_SIZE,
        image_size: Optional[Tuple[int, int]] = None,
        debug_input_output_dir: Optional[Path] = None,
    ):
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet
        self.thresholds = thresholds if thresholds is not None else [0.9, 0.95, 0.95]
        self.factor = factor
        self.min_face_magnitude = min_face_magnitude
        self.image_size = image_size
        self.debug_input_output_dir = debug_input_output_dir

    def predict(self, rgb_image: np.ndarray):
        if rgb_image.dtype != np.float32:
            rgb_image = rgb_image.astype(np.float32)
        if self.image_size:
            orig_h, orig_w, _ = rgb_image.shape
            resize_h, resize_w = self.image_size
            rgb_image = cv2.resize(
                rgb_image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR
            )

        t1, t2, t3 = self.thresholds
        _, boxes = stage_one(
            self.pnet,
            rgb_image,
            self.min_face_magnitude,
            self.factor,
            t1,
            debug_input_output_dir=self.debug_input_output_dir,
        )
        _, boxes = stage_two(
            self.rnet,
            rgb_image,
            boxes,
            t2,
            debug_input_output_dir=self.debug_input_output_dir,
        )
        _, boxes, landmarks = stage_three(
            self.onet,
            rgb_image,
            boxes,
            t3,
            debug_input_output_dir=self.debug_input_output_dir,
        )
        if self.image_size:
            shift_w = orig_w / resize_w
            shift_h = orig_h / resize_h
            boxes[:, 0] *= shift_w
            boxes[:, 1] *= shift_h
            boxes[:, 2] *= shift_w
            boxes[:, 3] *= shift_h
            landmarks[:, :5] *= shift_w
            landmarks[:, 5:] *= shift_h
        return boxes, landmarks

    @staticmethod
    def keras_predictors(
        data_format: str = "channels_last",
        debug_input_output_dir: Optional[Path] = None,
        **kwargs,
    ):
        from mtcnn_base import KerasPNet, KerasRNet, KerasONet

        return MTCNN(
            KerasPNet.default_model(
                data_format=data_format, debug_input_output_dir=debug_input_output_dir
            ),
            KerasRNet.default_model(
                data_format=data_format, debug_input_output_dir=debug_input_output_dir
            ),
            KerasONet.default_model(
                data_format=data_format, debug_input_output_dir=debug_input_output_dir
            ),
            debug_input_output_dir=debug_input_output_dir,
            **kwargs,
        )


def stage_one(
    pnet,
    image,
    min_size,
    factor,
    threshold,
    debug_input_output_dir: Optional[Path] = None,
):
    image = image.astype(np.float32)
    image = (image - 127.5) / 128.0
    height, width, _ = image.shape
    scales = compute_scales(min_size, factor, height, width)

    probs = []
    regs = []
    bboxs = []
    for scale_idx, scale in enumerate(
        scales
    ):  # TODO: we can reduce the number of scales and save some time
        height_scaled, width_scaled = compute_height_width_at_scale(
            scale, height, width
        )
        resized_image = cv2.resize(
            image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA
        )
        # print(resized_image.reshape(-1)[:10])
        # import sys

        # sys.exit(1)

        resized_image = resized_image.reshape((1, *resized_image.shape))
        prob, reg = pnet(resized_image)
        # print(prob.reshape(-1)[:10])
        # print(reg.reshape(-1)[:10])
        # import sys

        # sys.exit(1)

        input_prob = np.array(prob, copy=True)
        input_reg = np.array(reg, copy=True)

        prob = scipy.special.softmax(prob, axis=-1)

        (prob,) = prob
        (reg,) = reg
        prob, reg, bbox = generate_bounding_box(prob, reg, threshold, scale)

        # import pdb

        # pdb.set_trace()

        save_input_output(
            debug_input_output_dir,
            inputs=[input_prob, input_reg],
            input_prefixes=[
                f"generate-boxes_{scale_idx}_prob",
                f"generate-boxes_{scale_idx}_reg",
            ],
            outputs=[prob, reg, bbox],
            output_prefixes=[
                f"generate-boxes_{scale_idx}_output-prob",
                f"generate-boxes_{scale_idx}_output-reg",
                f"generate-boxes_{scale_idx}_output-boxes",
            ],
        )

        indices = nms_indices(bbox, prob, iou_threshold=0.5)
        prob = prob[indices]
        reg = reg[indices]
        bbox = bbox[indices]

        # import pdb

        # pdb.set_trace()

        # save_input_output(
        #     debug_input_output_dir,
        #     inputs=[bbox, prob],
        #     input_prefixes=[
        #         f"nms-indices_{scale_idx}_boxes",
        #         f"nms-indices_{scale_idx}_prob",
        #     ],
        #     outputs=[],
        # )

        probs.append(prob)
        regs.append(reg)
        bboxs.append(bbox)

    prob = np.concatenate(probs, axis=0)
    reg = np.concatenate(regs, axis=0)
    bbox = np.concatenate(bboxs, axis=0)

    indices = nms_indices(bbox, prob, iou_threshold=0.7)
    prob = prob[indices]
    reg = reg[indices]
    bbox = bbox[indices]

    inpt_box = np.array(bbox, copy=True)
    inpt_reg = np.array(reg, copy=True)

    bbox = regress_box(bbox, reg)
    outp_reg = np.array(bbox, copy=True)
    bbox = square_box(bbox)

    save_input_output(
        debug_input_output_dir,
        inputs=[inpt_box, inpt_reg],
        input_prefixes=[
            f"regress-and-square_input-box",
            f"regress-and-square_input-reg",
        ],
        outputs=[outp_reg, bbox],
        output_prefixes=[
            "regress-and-square_output-reg-boxes",
            "regress-and-square_output-reg-and-sq-boxes",
        ],
    )

    return prob, bbox


def stage_two(
    rnet, image, bbox, threshold, debug_input_output_dir: Optional[Path] = None,
):
    image = (image - 127.5) / 128
    target_size = (24, 24)

    resized_thumbnails = _crop_thumbnails(image, bbox, target_size)

    # These transposes should live at the network level
    prob, reg = rnet(resized_thumbnails)

    prob = prob[:, 1]

    mask = prob > threshold
    prob = prob[mask]
    reg = reg[mask]
    bbox = bbox[mask]

    idxs = nms_indices(bbox, prob, iou_threshold=0.7)
    prob = prob[idxs]
    reg = reg[idxs]
    bbox = bbox[idxs]

    bbox = regress_box(bbox, reg)
    bbox = square_box(bbox)
    return prob, bbox


def stage_three(
    onet, image, bbox, threshold, debug_input_output_dir: Optional[Path] = None,
):
    image = (image - 127.5) / 128
    target_size = (48, 48)

    resized_thumbnails = _crop_thumbnails(image, bbox, target_size)

    # These transposes should live at the network level
    prob, reg, landmarks = onet(resized_thumbnails)
    prob = prob[:, 1]

    mask = prob > threshold
    prob = prob[mask]
    reg = reg[mask]
    bbox = bbox[mask]
    landmarks = np.clip(
        landmarks[mask], 0.0, 1.0
    )  # nothing forces the values out of the network to between 0.0 and 1.0

    # landmarks are normalized values
    # so we multiply by the box with, height and shift by the xmin, ymin of the box
    wh = bbox[..., 2:] - bbox[..., :2]
    landmarks[:, :5] *= wh[:, 0:1]
    landmarks[:, :5] += bbox[:, 0:1]
    landmarks[:, 5:] *= wh[:, 1:2]
    landmarks[:, 5:] += bbox[:, 1:2]

    bbox = regress_box(bbox, reg)

    idxs = nms_indices(bbox, prob, iou_threshold=0.6)
    prob = prob[idxs]
    reg = reg[idxs]
    bbox = bbox[idxs]
    landmarks = landmarks[idxs]
    return prob, bbox, landmarks


def generate_bounding_box(prob, reg, threshold, scale):
    """Create boxes from net output

    Assume input is 1 image at a time.
    """
    stride = 2
    cell_size = 12

    prob = prob[..., 1]
    mask = prob > threshold
    y, x = np.where(mask)
    indices = np.stack([x, y], axis=1)
    bbox = np.concatenate(
        ((indices * stride + 1) / scale, (indices * stride + cell_size) / scale),
        axis=1,
    ).astype(np.float32)
    prob = prob[mask]
    reg = reg[mask]
    return prob, reg, bbox


def compute_scales(min_size, factor, height, width):
    scale = 12.0 / min_size
    cur_side = min(height, width) * scale
    scales = []
    while cur_side >= 12:
        scales.append(scale)
        scale *= factor
        cur_side *= factor
    return scales


def compute_height_width_at_scale(scale, height, width):
    width_scaled = int(np.ceil(width * scale))
    height_scaled = int(np.ceil(height * scale))
    return height_scaled, width_scaled


def regress_box(bbox, reg):
    wh = bbox[..., 2:] - bbox[..., :2]
    bbox[..., :2] += wh * reg[..., :2]
    bbox[..., 2:] += wh * reg[..., 2:]
    return bbox


def square_box(bbox):
    wh = bbox[..., 2:] - bbox[..., :2]
    max_side = np.max(wh, axis=1).reshape((-1, 1))
    delta = (wh - max_side) * 0.5
    bbox[..., :2] += delta
    bbox[..., 2:] -= delta
    return bbox


def _crop_thumbnails(image, bbox, target_size):
    resized_thumbnails = np.empty((len(bbox), *target_size, 3), dtype=np.float32)
    bbox_int = bbox.astype(np.int32)
    for i, (xmin, ymin, xmax, ymax) in enumerate(bbox_int):
        thumbnail = image[ymin:ymax, xmin:xmax, :]
        if thumbnail.size != 0:
            resized_thumbnails[i, ...] = cv2.resize(  # Can I use dst here?
                thumbnail, target_size, interpolation=cv2.INTER_AREA
            )
    return resized_thumbnails


def nms_indices(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> List[int]:
    """Non maximal surpression of bounding boxes in numpy.

    Args:
        boxes: we assume format [xmin, ymin, xmax, ymax]
        scores: box score to rank boxes
        iou_threshold:

    Returns
        List[int]: indices of boxes and scores to keep

    Notes:
        Useful Resource:
        https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    """
    keep = []

    xmins = boxes[:, 0]
    ymins = boxes[:, 1]
    xmaxs = boxes[:, 2]
    ymaxs = boxes[:, 3]

    area = (xmaxs - xmins) * (ymaxs - ymins)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        i = idxs[-1]
        keep.append(i)

        xxmins = np.maximum(xmins[i], xmins[idxs[:-1]])
        yymins = np.maximum(ymins[i], ymins[idxs[:-1]])
        xxmaxs = np.minimum(xmaxs[i], xmaxs[idxs[:-1]])
        yymaxs = np.minimum(ymaxs[i], ymaxs[idxs[:-1]])

        w = np.maximum(0, xxmaxs - xxmins)
        h = np.maximum(0, yymaxs - yymins)

        intersection_area = w * h
        overlap = intersection_area / (area[i] + area[idxs[:-1]] - intersection_area)

        idxs = idxs[:-1]
        idxs = idxs[overlap < iou_threshold]

    return keep
