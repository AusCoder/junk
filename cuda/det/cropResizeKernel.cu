// blockSize will be {1024, 1, 1}
// gridSize...
__global__ void
cropResizeCHWKernel(const float *image, int imageWidth, int imageHeight,
                    int depth, const float *boxes, int boxesSize, int cropWidth,
                    int cropHeight, float *croppedResizedImages,
                    int croppedResizedImagesSize // == boxesSize * depth *
                                                 // cropWidth * cropHeight
) {
  // Assume that the depth is 3 for now

  const float extrapolationValue = 0.0f;
  const int batch = 1;

  // Each thread will loop and write to certain Idx in croppedResizedImages
  for (int outIdx = threadIdx.x + blockIdx.x * blockDim.x;
       outIdx < croppedResizedImagesSize; outIdx += blockDim.x * gridDim.x) {
    int idx = outIdx;
    const int x = idx % cropWidth;
    idx /= cropWidth;
    const int y = idx % cropHeight;
    idx /= cropHeight;
    const int depthIdx = idx % depth;
    const int boxIdx = idx / depth;

    const float y1 = boxes[boxIdx * 4];
    const float x1 = boxes[boxIdx * 4 + 1];
    const float y2 = boxes[boxIdx * 4 + 2];
    const float x2 = boxes[boxIdx * 4 + 3];

    const int batchIdx = boxIdx / boxesSize;
    if (batchIdx < 0 || batchIdx >= batch) {
      continue;
    }

    const float heightScale =
        (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) : 0;
    const float widthScale =
        (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;

    const float inY = (cropHeight > 1)
                          ? y1 * (imageHeight - 1) + y * heightScale
                          : 0.5 * (y1 + y2) * (imageHeight - 1);
    if (inY < 0 || inY > imageHeight - 1) {
      croppedResizedImages[outIdx] = extrapolationValue;
      continue;
    }

    const float inX = (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale
                                      : 0.5 * (x1 + x2) * (imageWidth - 1);
    if (inX < 0 || inX > imageWidth - 1) {
      croppedResizedImages[outIdx] = extrapolationValue;
      continue;
    }

    const int topYIndex = floorf(inY);
    const int bottomYIndex = ceilf(inY);
    const float yLerp = inY - topYIndex;
    const int leftXIndex = floorf(inX);
    const int rightXIndex = ceilf(inX);
    const float xLerp = inX - leftXIndex;

    const float topLeft(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + topYIndex) *
                  imageWidth +
              leftXIndex]));
    const float topRight(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + topYIndex) *
                  imageWidth +
              rightXIndex]));
    const float bottomLeft(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + bottomYIndex) *
                  imageWidth +
              leftXIndex]));
    const float bottomRight(static_cast<float>(
        image[((batchIdx * depth + depthIdx) * imageHeight + bottomYIndex) *
                  imageWidth +
              rightXIndex]));

    const float top = topLeft + (topRight - topLeft) * xLerp;
    const float bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
    croppedResizedImages[outIdx] = top + (bottom - top) * yLerp;
  }
}

__global__ void
cropResizeHWCKernel(const float *image, int imageWidth, int imageHeight,
                    int depth, const float *boxes, int boxesSize, int cropWidth,
                    int cropHeight, float *croppedResizedImages,
                    int croppedResizedImagesSize // == boxesSize * cropWidth *
                                                 // cropHeight * depth
) {
  // Assume that the depth is 3 for now

  const float extrapolationValue = 0.0f;
  const int batch = 1;

  // Each thread will loop and write to certain Idx in croppedResizedImages
  for (int outIdx = threadIdx.x + blockIdx.x * blockDim.x;
       outIdx < croppedResizedImagesSize; outIdx += blockDim.x * gridDim.x) {
    // int idx = outIdx;
    // const int x = idx % cropWidth;
    // idx /= cropWidth;
    // const int y = idx % cropHeight;
    // idx /= cropHeight;
    // const int depthIdx = idx % depth;
    // const int boxIdx = idx / depth;
    int idx = outIdx;
    const int depthIdx = idx % depth;
    idx /= depth;
    const int x = idx % cropWidth;
    idx /= cropWidth;
    const int y = idx % cropHeight;
    idx /= cropHeight;
    const int boxIdx = idx;

    const float y1 = boxes[boxIdx * 4];
    const float x1 = boxes[boxIdx * 4 + 1];
    const float y2 = boxes[boxIdx * 4 + 2];
    const float x2 = boxes[boxIdx * 4 + 3];

    const int batchIdx = boxIdx / boxesSize;
    if (batchIdx < 0 || batchIdx >= batch) {
      continue;
    }

    const float heightScale =
        (cropHeight > 1) ? (y2 - y1) * (imageHeight - 1) / (cropHeight - 1) : 0;
    const float widthScale =
        (cropWidth > 1) ? (x2 - x1) * (imageWidth - 1) / (cropWidth - 1) : 0;

    const float inY = (cropHeight > 1)
                          ? y1 * (imageHeight - 1) + y * heightScale
                          : 0.5 * (y1 + y2) * (imageHeight - 1);
    if (inY < 0 || inY > imageHeight - 1) {
      croppedResizedImages[outIdx] = extrapolationValue;
      continue;
    }

    const float inX = (cropWidth > 1) ? x1 * (imageWidth - 1) + x * widthScale
                                      : 0.5 * (x1 + x2) * (imageWidth - 1);
    if (inX < 0 || inX > imageWidth - 1) {
      croppedResizedImages[outIdx] = extrapolationValue;
      continue;
    }

    const int topYIndex = floorf(inY);
    const int bottomYIndex = ceilf(inY);
    const float yLerp = inY - topYIndex;
    const int leftXIndex = floorf(inX);
    const int rightXIndex = ceilf(inX);
    const float xLerp = inX - leftXIndex;

    const float topLeft(static_cast<float>(
        image[((batchIdx * imageHeight + topYIndex) * imageWidth + leftXIndex) *
                  depth +
              depthIdx]));
    const float topRight(static_cast<float>(
        image[((batchIdx * imageHeight + topYIndex) * imageWidth +
               rightXIndex) *
                  depth +
              depthIdx]));
    const float bottomLeft(static_cast<float>(
        image[((batchIdx * imageHeight + bottomYIndex) * imageWidth +
               leftXIndex) *
                  depth +
              depthIdx]));
    const float bottomRight(static_cast<float>(
        image[((batchIdx * imageHeight + bottomYIndex) * imageWidth +
               rightXIndex) *
                  depth +
              depthIdx]));

    const float top = topLeft + (topRight - topLeft) * xLerp;
    const float bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
    croppedResizedImages[outIdx] = top + (bottom - top) * yLerp;
  }
}

void cropResizeCHW(const float *image, int imageWidth, int imageHeight,
                   int depth, const float *boxes, int boxesSize, int cropWidth,
                   int cropHeight, float *croppedResizedImages,
                   int croppedResizedImagesSize // == boxesSize * cropWidth *
                                                // cropHeight * depth
) {
  const int block = 1024;
  const int grid = (croppedResizedImagesSize + block - 1) / block;

  cropResizeCHWKernel<<<grid, block>>>(
      image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
      cropHeight, croppedResizedImages, croppedResizedImagesSize);
}

void cropResizeCHW(const float *image, int imageWidth, int imageHeight,
  int depth, const float *boxes, int boxesSize, int cropWidth,
  int cropHeight, float *croppedResizedImages,
  int croppedResizedImagesSize, cudaStream_t *stream){
    const int block = 1024;
    const int grid = (croppedResizedImagesSize + block - 1) / block;

    cropResizeCHWKernel<<<grid, block, 0, *stream>>>(
        image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
        cropHeight, croppedResizedImages, croppedResizedImagesSize);
  }

void cropResizeHWC(const float *image, int imageWidth, int imageHeight,
    int depth, const float *boxes, int boxesSize, int cropWidth,
    int cropHeight, float *croppedResizedImages,
    int croppedResizedImagesSize // == boxesSize * cropWidth *
                                 // cropHeight * depth
) {
const int block = 1024;
const int grid = (croppedResizedImagesSize + block - 1) / block;

cropResizeHWCKernel<<<grid, block>>>(
image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
cropHeight, croppedResizedImages, croppedResizedImagesSize);
}

void cropResizeHWC(const float *image, int imageWidth, int imageHeight,
  int depth, const float *boxes, int boxesSize, int cropWidth,
  int cropHeight, float *croppedResizedImages,
  int croppedResizedImagesSize, cudaStream_t *stream) {
    const int block = 1024;
    const int grid = (croppedResizedImagesSize + block - 1) / block;

    cropResizeHWCKernel<<<grid, block, 0, *stream>>>(
    image, imageWidth, imageHeight, depth, boxes, boxesSize, cropWidth,
    cropHeight, croppedResizedImages, croppedResizedImagesSize);
    }