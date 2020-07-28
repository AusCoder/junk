#ifndef _CROP_RESIZE_KERNEL_H
#define _CROP_RESIZE_KERNEL_H

void cropResizeCHW(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize);

void cropResizeHWC(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize);

#endif