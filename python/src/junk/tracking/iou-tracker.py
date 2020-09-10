import sys
import time
from collections import namedtuple
from typing import Any, Iterable, List, Optional

import cv2
import numpy as np
import matplotlib.cm as cm
from scipy import optimize


KEY_ESC = 27
IMSHOW_SLEEP_TIME = 500

COLORS = (cm.jet(np.linspace(0, 1.0, 11))[:, :3] * 256).astype(int)
BOX_THICKNESS = 2
MEASUREMENT_COLOR = (0, 255, 0)
TRACK_COLORS = COLORS

IOU_THRESHOLD = 0.1
TRACKER_MAX_AGE = 3


class IoUTracker:
    count = 0

    def __init__(self, initial_box: np.ndarray) -> None:
        self.box = initial_box
        self.idx = IoUTracker.count
        IoUTracker.count += 1

        self.color = tuple(
            int(x) % 256 for x in TRACK_COLORS[self.idx % len(TRACK_COLORS)]
        )
        self.predictions_since_update = 0

    def predict(self):
        # Doesn't do anything!
        # There is no forward state propogation
        self.predictions_since_update += 1

    def update(self, measurement: np.ndarray) -> None:
        self.box = measurement
        self.predictions_since_update = 0


def run_tracking(measurement_type: str) -> None:
    image_arr = np.ones((720, 1280, 3), dtype=np.uint8) * 100

    # Simulated measurements
    if measurement_type == "slide":
        all_measurements = get_sliding_boxes_measurements(image_arr)
    elif measurement_type == "circle":
        all_measurements = get_circle_measurements(image_arr, acceleration=1.0)
    elif measurement_type == "circle-accel":
        all_measurements = get_circle_measurements(image_arr, acceleration=1.1)
    elif measurement_type == "collide":
        all_measurements = get_collide_measurements(image_arr)
    else:
        raise ValueError(
            f"Invalid measurement type: {measurement_type}. Expected one of: slide, circle, circle-accel, collide"
        )

    # Current set of trackers
    trackers = []

    for measurements in all_measurements:
        current_frame = np.array(image_arr, copy=True)
        # Draw current tracker state
        for tracker in trackers:
            current_frame = draw_bounding_boxes(
                current_frame, [tracker.box], color=tracker.color
            )
        # Draw measurement
        current_frame = draw_bounding_boxes(
            current_frame, measurements, color=MEASUREMENT_COLOR
        )
        # Update trackers
        trackers = predict_match_create_update_and_kill(measurements, trackers)
        # Show the result
        show_image(current_frame)


def get_sliding_boxes_measurements(image_arr: np.ndarray) -> List[np.ndarray]:
    all_measurements = [
        np.array([[x, 100, x + 100, 200], [x, 500, x + 100, 550]])
        for x in range(0, 1000, 50)
    ]
    start_time = 2
    for n in range(len(all_measurements) // 2):
        all_measurements[start_time + n] = np.concatenate(
            (
                all_measurements[start_time + n],
                np.array([[50 * n + 10, 300, 50 * n + 40, 400]]),
            ),
            axis=0,
        )
    return all_measurements


def get_circle_measurements(
    image_arr: np.ndarray, acceleration: float = 1.0
) -> List[np.ndarray]:
    measurements_length = 30
    angular_velocity = np.pi / 24
    box_size = 100
    img_h, img_w, _ = image_arr.shape
    center_x, center_y = img_w // 2, img_h // 2
    radius = min(img_h, img_w) // 2 - box_size

    all_measurements = []
    for i in range(measurements_length):
        top_left_points = [
            (
                center_x + radius * np.sin(np.pi * 2 * j / 3 + i * angular_velocity),
                center_y + radius * np.cos(np.pi * 2 * j / 3 + i * angular_velocity),
            )
            for j in range(3)
        ]
        measurements = np.array(
            [[x, y, x + box_size, y + box_size] for x, y in top_left_points]
        ).astype(int)
        all_measurements.append(measurements)
        angular_velocity *= acceleration
    return all_measurements


def get_collide_measurements(image_arr: np.ndarray) -> List[np.ndarray]:
    img_h, img_w, _ = image_arr.shape
    measurements_length = 30
    box_size = 100
    velocity = img_w // measurements_length

    all_measurements = []
    for i in range(measurements_length):
        left_position = i * velocity
        top_position = img_h // 2
        measurements = np.array(
            [
                [
                    left_position,
                    top_position,
                    left_position + box_size,
                    top_position + box_size,
                ],
                [
                    img_w - box_size - (left_position + box_size),
                    top_position,
                    img_w - box_size - left_position,
                    top_position + box_size,
                ],
            ]
        )
        all_measurements.append(measurements)
    return all_measurements


def predict_match_create_update_and_kill(
    measurements: np.ndarray, trackers: List[IoUTracker]
) -> List[IoUTracker]:
    # Tracker propogation
    for tracker in trackers:
        tracker.predict()

    # Cost calculation and assignment
    tracker_arr = np.empty((0, 4), dtype=float)
    if trackers:
        tracker_arr = np.stack([tracker.box for tracker in trackers])
    cost_matrix = calculate_cost_matrix(measurements, tracker_arr)
    measurement_assignments, tracker_assignments = optimize.linear_sum_assignment(
        cost_matrix
    )
    # We just want assignments where the IoU is > 0
    valid_assignments = [
        (measurement_idx, tracker_idx)
        for measurement_idx, tracker_idx in zip(
            measurement_assignments, tracker_assignments
        )
        if cost_matrix[measurement_idx, tracker_idx] < 0
    ]

    # Update tracker based on assignments
    for measurement_idx, tracker_idx in valid_assignments:
        trackers[tracker_idx].update(measurements[measurement_idx])

    # Create new trackers
    valid_measurement_assignments = set(m for m, _ in valid_assignments)
    new_trackers = [
        IoUTracker(measurement)
        for i, measurement in enumerate(measurements)
        if i not in valid_measurement_assignments
    ]
    # Evict old trackers
    live_trackers = [
        tracker
        for tracker in trackers + new_trackers
        if tracker.predictions_since_update < TRACKER_MAX_AGE
    ]
    return live_trackers


def calculate_cost_matrix(measurements: np.ndarray, trackers: np.ndarray) -> np.ndarray:
    assert measurements.ndim == 2
    assert trackers.ndim == 2

    cost_matrix = np.empty((len(measurements), len(trackers)), dtype=float)
    for i, measurement in enumerate(measurements):
        cost_matrix[i, :] = -calculate_iou(measurement, trackers)
    return cost_matrix


def calculate_iou(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    if b2.size == 0:
        return np.empty((0), dtype=float)
    if b2.ndim == 1:
        b2 = b2.reshape((1, -1))
    assert b1.shape == (4,)
    assert b2.ndim == 2

    x_min = np.maximum(b1[0], b2[:, 0])
    y_min = np.maximum(b1[1], b2[:, 1])
    x_max = np.minimum(b1[2], b2[:, 2])
    y_max = np.minimum(b1[3], b2[:, 3])

    intersection = np.maximum(0, x_max - x_min) * np.maximum(0, y_max - y_min)
    area_b1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area_b2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = np.maximum(1e-5, ((area_b1 + area_b2) - intersection))
    return intersection / union


def draw_bounding_boxes(
    image_arr: np.ndarray, bounding_boxes: Iterable[np.ndarray], *, color: Any
) -> np.ndarray:
    new_image_arr = np.array(image_arr, copy=True)
    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(
            new_image_arr,
            (x_min, y_min),
            (x_max, y_max),
            color=color,
            thickness=BOX_THICKNESS,
        )
    return new_image_arr


def draw_and_show_bounding_boxes(
    image_arr: np.ndarray,
    bounding_boxes: Iterable[np.ndarray],
    *,
    color: Any,
    should_wait: bool = False,
    sleep_time_ms: int = IMSHOW_SLEEP_TIME,
) -> None:
    new_image_arr = draw_bounding_boxes(image_arr, bounding_boxes, color=color)
    show_image(new_image_arr, should_wait=should_wait, sleep_time_ms=sleep_time_ms)


def show_image(
    image_arr: np.ndarray,
    *,
    should_wait: bool = False,
    sleep_time_ms: int = IMSHOW_SLEEP_TIME,
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
    measurement_type = sys.argv[1] if len(sys.argv) > 1 else "circle"
    run_tracking(measurement_type)
    cv2.destroyAllWindows()
