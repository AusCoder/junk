import time
from collections import namedtuple
from typing import Any, Iterable, Optional

import cv2
import numpy as np

KEY_ESC = 27

BOX_THICKNESS = 2
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

IOU_THRESHOLD = 0.1


PBox = namedtuple("PBox", ["x_min", "y_min", "x_max", "y_max"])


class IoUTracker:
    def __init__(self, initial_box: PBox) -> None:
        self.box = initial_box

    def update(self, measurement: PBox) -> None:
        self.box = measurement


def iou(b1: PBox, b2: PBox) -> float:
    pass


def run_tracking():
    image_arr = np.ones((720, 1280, 3), dtype=np.uint8) * 100

    all_measurements = [[PBox(x, 100, x + 100, 200)] for x in range(0, 1000, 50)]
    tracker: Optional[IoUTracker] = None

    for i, measurements in enumerate(all_measurements):
        current_frame = np.array(image_arr, copy=True)
        # Draw current tracker state
        if tracker:
            current_frame = draw_bounding_boxes(
                current_frame, [tracker.box], color=COLOR_RED
            )
        # Draw measurement
        current_frame = draw_bounding_boxes(
            current_frame, measurements, color=COLOR_GREEN
        )
        # Update trackers
        (measurement,) = measurements
        if tracker is None:
            tracker = IoUTracker(measurement)
        else:
            # TODO: idea is to only update if the IOU is positive
            if iou(measurement, tracker.box) > IOU_THRESHOLD:
                tracker.update(measurement)
        show_image(current_frame)


def draw_bounding_boxes(
    image_arr: np.ndarray, bounding_boxes: Iterable[PBox], *, color: Any = COLOR_GREEN
) -> np.ndarray:
    new_image_arr = np.array(image_arr, copy=True)
    for box in bounding_boxes:
        cv2.rectangle(
            new_image_arr,
            (box.x_min, box.y_min),
            (box.x_max, box.y_max),
            color=color,
            thickness=BOX_THICKNESS,
        )
    return new_image_arr


def draw_and_show_bounding_boxes(
    image_arr: np.ndarray,
    bounding_boxes: Iterable[PBox],
    *,
    color: Any = COLOR_GREEN,
    should_wait: bool = False,
    sleep_time_ms: int = 500,
) -> None:
    new_image_arr = draw_bounding_boxes(image_arr, bounding_boxes, color=color)
    show_image(new_image_arr, should_wait=should_wait, sleep_time_ms=sleep_time_ms)


def show_image(
    image_arr: np.ndarray, *, should_wait: bool = False, sleep_time_ms: int = 500
) -> None:
    cv2.imshow("window", image_arr)

    if should_wait:
        while True:
            k = cv2.waitKey()
            if k in [ord("q"), KEY_ESC]:
                break
    else:
        cv2.waitKey(sleep_time_ms)


if __name__ == "__main__":
    # image_arr = np.ones((720, 1280, 3), dtype=np.uint8) * 100
    # draw_and_show_bounding_boxes(image_arr, [PBox(50, 50, 300, 300)])
    run_tracking()
    cv2.destroyAllWindows()
