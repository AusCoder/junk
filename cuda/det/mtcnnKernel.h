#ifndef _MTCNN_KERNEL_H
#define _MTCNN_KERNEL_H

#include <cuda_runtime.h>

void normalizePixels(float *image, size_t imageSize, cudaStream_t *stream);

void denormalizePixels(float *image, size_t imageSize, cudaStream_t *stream);

void debugPrintVals(float *image, size_t numVals, size_t offset,
                    cudaStream_t *stream);

#endif