"""Experiments with fourier transforms.

Good explanation from here:
https://homepages.inf.ed.ac.uk/rbf/HIPR2/fourier.htm
"""

from pathlib import Path

import cv2
import numpy as np

ESC_KEY = 27


def calculate_fft(bgr_image: np.ndarray, threshold: int = 0) -> np.ndarray:
    grey = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    grey = grey.astype(np.float32)
    f = np.fft.fft2(grey)
    fshift = np.fft.fftshift(f)
    # XXX: Some kind of thresholding is needed
    # fshift[fshift < threshold] = 0
    mag = 20 * np.log(np.abs(fshift))
    return f, fshift, mag


def main():
    image_path = "fft-data/stp1.gif"
    # cv2.imread doesn't work with gifs
    cap = cv2.VideoCapture(image_path)
    ret, image = cap.read()
    assert ret
    cap.release()

    fft, fftshift, mag = calculate_fft(image, 0)
    cv2.imshow("image", image)
    cv2.imshow("fft-magnitude", mag.astype(np.uint8))

    while True:
        k = cv2.waitKey(0)
        if k & 0xFF in [ord("q"), ESC_KEY]:
            break


if __name__ == "__main__":
    main()
