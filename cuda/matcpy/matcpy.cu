#include <cmath>
#include <unistd.h>
#include <iostream>
#include <cuda_runtime_api.h>
#define N 257

using namespace std;

// Assumes that the block and grid sizes divide the width of array
// Doesn't to bounds checking
// Potential trouble!
// Use with:    dim3 block(32, 32); dim3 grid(8, 8);
__global__ void matcpyBad(const float *in, float *out) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int width = blockDim.x * gridDim.x;
    out[x + y * width] = in[x + y * width];
}

// Use with:    dim3 block(32, 32); dim3 grid(8, 2);
__global__ void matcpyBadWithLoop(const float *in, float *out) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int width = blockDim.x * gridDim.x;
    for (int j = 0; j < 4 * blockDim.y * gridDim.y; j += blockDim.y * gridDim.y) {
        int idx = (y + j) * width + x;
        out[idx] = in[idx];
    }
}

// Works with different blocks sizes
__global__ void matcpy(const float *in, int size, float *out) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * blockDim.x * gridDim.x;
    if (idx < size) {
        out[idx] = in[idx];
    }
}

__global__ void matcpyWithLoop(const float *in, int size, float *out) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int width = blockDim.x * gridDim.x;
    for (int j = 0; j < 4 * blockDim.y * gridDim.y; j += blockDim.y * gridDim.y) {
        int idx = (y + j) * width + x;
        if (idx < size) {
            out[idx] = in[idx];
        }
    }
}

int main() {
    size_t size = N * N * sizeof(float);
    dim3 block(32, 32);
    dim3 grid((N - 1) / block.x + 1, (N - 1) / (4 * block.y) + 1);

    float in[N * N];
    float out[N * N] = {0.0f};
    for (int i = 0; i < N * N; i++) {
        in[i] = i;
    }

    float *dIn;
    float *dOut;

    cudaMalloc((void **)&dIn, size);
    cudaMalloc((void **)&dOut, size);
    cudaMemcpy(dIn, in, size, cudaMemcpyHostToDevice);
    matcpyWithLoop<<<grid, block>>>(dIn, N * N, dOut);
    cudaMemcpy(out, dOut, size, cudaMemcpyDeviceToHost);

    bool areEqual = true;
    for (int i = 0; i < N * N; i++) {
        areEqual &= abs(in[i] - out[i]) < 0.00001;
    }
    if (areEqual) {
        cout << "copy successful" << endl;
    } else {
        cout << "copy failed" << endl;
    }

    // usleep(1000000);

    return 0;
}
