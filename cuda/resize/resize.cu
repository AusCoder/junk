/*
This seems to work.
- Try with a real image
- Do I need to shift by a bit in int(ox * wScale)?
*/

#include "resize.h"

__global__ void resizeNearestKernelNCHW(float const* in, NCHWSize inputSize, float *out, NCHWSize outputSize)
{
    int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    int z0 = blockIdx.z;
    for (int batch = z0; batch < outputSize.batch * outputSize.channels; batch += gridDim.z)
    {
        for (int oy = y0; oy < outputSize.height; oy += blockDim.y * gridDim.y)
        {
            for (int ox = x0; ox < outputSize.width; ox += blockDim.x * gridDim.x)
            {
                float wScale = (float)inputSize.width / (float)outputSize.width;
                float hScale = (float)inputSize.height / (float)outputSize.height;
                int ix = int(ox * wScale);
                int iy = int(oy * hScale);
                int inIdx = batch * inputSize.height * inputSize.width + iy * inputSize.width + ix;
                int outIdx = batch * outputSize.height * outputSize.width + oy * outputSize.width + ox;
                out[outIdx] = in[inIdx];
            }
        }
    }
}

void resizeNearest(dim3 grid, dim3 block,
    float const* in, NCHWSize inputSize, float *out, NCHWSize outputSize)
{
    resizeNearestKernelNCHW<<<grid, block>>>(in, inputSize, out, outputSize);
}

// __global__ void resizeBilinearKernelNCHW(float const* in, NCHWSize inputSize, float *out, NCHWSize outputSize)
// {}

// void resizeBilinear(dim3 grid, dim3 block,
//     float const* in, NCHWSize inputSize, float *out, NCHWSize outputSize)
// {

// }
