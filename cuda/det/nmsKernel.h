#ifndef _NMS_KERNEL_H
#define _NMS_KERNEL_H

#include <cuda_runtime.h>

void nmsSimple(float *boxes, size_t boxesSize, float *outBoxes,
               size_t outBoxesSize, float iouThreshold);

#endif