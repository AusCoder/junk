#pragma once

#include <cuda_runtime_api.h>

class NCHWSize {
    public:
        NCHWSize(unsigned int b, unsigned int c, unsigned int h, unsigned int w): batch(b), channels(c), height(h), width(w) {}

        inline unsigned int volume() {
            return batch * channels * height * width;
        }

        unsigned int batch;
        unsigned int channels;
        unsigned int height;
        unsigned int width;
};

void resizeNearest(dim3 grid, dim3 block,
    float const* in, NCHWSize inputSize, float *out, NCHWSize outputSize);
