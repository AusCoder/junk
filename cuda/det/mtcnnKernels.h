#ifndef _MTCNN_KERNEL_H
#define _MTCNN_KERNEL_H

#include <cuda_runtime.h>

/**
 * This is nms without accounting for a probability score,
 * ie no box sorting takes place at the beginning.
 * I should add this!
 */
void nmsSimple(float *boxes, size_t boxesSize, float *outBoxes,
               size_t outBoxesSize, float iouThreshold);

void normalizePixels(float *image, size_t imageSize, cudaStream_t *stream);

void denormalizePixels(float *image, size_t imageSize, cudaStream_t *stream);

void debugPrintVals(float *image, size_t numVals, size_t offset,
                    cudaStream_t *stream);

void cropResizeCHW(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize);

void cropResizeCHW(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize, cudaStream_t *stream);

void cropResizeHWC(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize);

void cropResizeHWC(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize, cudaStream_t *stream);

/**
 * Assumes outIndices has size >= maxOutIndices
 * Assumes outBbox has size equal to 4 * outIndices size
 *
 * rename maxOutIndices to maxOutIndex? or outIndicesSize?
 */
void generateBoxesWithoutSoftmax(float *prob, int probWidth, int probHeight,
                                 int *outIndices, float *outBboxes,
                                 int maxOutIndices, float threshold,
                                 float scale, cudaStream_t *stream);

#endif