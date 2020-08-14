#ifndef _MTCNN_KERNEL_H
#define _MTCNN_KERNEL_H

#include "deviceMemory.hpp"
#include <cuda_runtime.h>

/**
 * This is nms without accounting for a probability score,
 * ie no box sorting takes place at the beginning.
 * I should add this!
 */
void nmsSimple(float *boxes, size_t boxesSize, float *outBoxes,
               size_t outBoxesSize, float iouThreshold);

/**
 * This will start writing to outProb, outReg, outBoxes at
 * *(outBoxesPosition.get()) and then update outBoxesPosition with the number of
 * boxes written.
 */
void nms(DevicePtr<float> prob, DevicePtr<float> reg, DevicePtr<float> boxes,
         DevicePtr<int> boxesCount, DevicePtr<int> sortIndices,
         DevicePtr<float> outProb, DevicePtr<float> outReg,
         DevicePtr<float> outBoxes, DevicePtr<int> outBoxesPosition,
         float iouThreshold, cudaStream_t &stream);

void normalizePixels(DevicePtr<float> image, cudaStream_t &stream);

void denormalizePixels(float *image, size_t imageSize, cudaStream_t *stream);

void debugPrintVals(float *image, size_t numVals, size_t offset,
                    cudaStream_t *stream);

void debugPrintVals(int *values, size_t numVals, size_t offset,
                    cudaStream_t *stream);

void cropResizeCHW(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize);

void cropResizeCHW(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize, cudaStream_t *stream);

void cropResizeHWC(DevicePtr<float> image, int imageWidth, int imageHeight,
                   int depth, DevicePtr<float> boxes, int boxesSize,
                   int cropWidth, int cropHeight,
                   DevicePtr<float> croppedResizedImages,
                   int croppedResizedImagesSize);

void cropResizeHWC(DevicePtr<float> image, int imageWidth, int imageHeight,
                   int depth, DevicePtr<float> boxes, int boxesSize,
                   int cropWidth, int cropHeight,
                   DevicePtr<float> croppedResizedImages,
                   int croppedResizedImagesSize, cudaStream_t &stream);

/**
 * Assumes outIndices has size >= maxOutIndices
 * Assumes outBbox has size equal to 4 * outIndices size
 *
 * rename maxOutIndices to maxOutIndex? or outIndicesSize?
 */
void generateBoxesWithoutSoftmaxIndices(float *prob, int probWidth,
                                        int probHeight, int *outIndices,
                                        float *outBboxes, int maxOutIndices,
                                        float threshold, float scale,
                                        cudaStream_t *stream);

void generateBoxesWithoutSoftmax(float *prob, int probWidth, int probHeight,
                                 float *reg, int regWidth, int regHeight,
                                 float *outProb, float *outReg,
                                 float *outBboxes, int *outBoxesCount,
                                 int maxOutputBoxes, float threshold,
                                 float scale, cudaStream_t *stream);

/**
 * Modifies boxes inplace
 *
 * Make boxexCount a size_t?
 */
void regressAndSquareBoxes(DevicePtr<float> boxes, DevicePtr<float> reg,
                           DevicePtr<int> boxesCount, bool shouldSquare,
                           cudaStream_t &stream);

void scalarMult(DevicePtr<float> p, size_t pSize, float value,
                cudaStream_t &stream);

#endif
