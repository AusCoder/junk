#include <iostream>
#include "resize.h"

using namespace std;

int main()
{
    NCHWSize inputSize{1, 3, 20, 20};
    NCHWSize outputSize{1, 3, 13, 34};

    dim3 block(32, 16);
    dim3 grid((outputSize.width - 1) / block.x + 1, (outputSize.height - 1) / block.y + 1, outputSize.batch * outputSize.channels);

    float in[inputSize.volume()];
    float out[outputSize.volume()];
    float *dIn;
    float *dOut;

    for (unsigned int i = 0; i < inputSize.volume(); i++) {
        in[i] = i;
    }

    size_t inSizeBs = inputSize.volume() * sizeof(float);
    size_t outSizeBs = outputSize.volume() * sizeof(float);
    cudaMalloc((void **)&dIn, inSizeBs);
    cudaMalloc((void **)&dOut, outSizeBs);
    cudaMemcpy(dIn, in, inSizeBs, cudaMemcpyHostToDevice);

    resizeNearest(grid, block, dIn, inputSize, dOut, outputSize);

    cudaMemcpy(out, dOut, outSizeBs, cudaMemcpyDeviceToHost);

    for (unsigned int i = 0; i < inputSize.height; i++) {
        for (unsigned int j = 0; j < inputSize.width; j++) {
            cout << in[i * inputSize.width + j] << " ";
        }
        cout << endl;
    }

    for (unsigned int i = 0; i < outputSize.height; i++) {
        for (unsigned int j = 0; j < outputSize.width; j++) {
            cout << out[i * outputSize.width + j] << " ";
        }
        cout << endl;
    }

    // bool match = true;
    // for (int i = 0; i < inputSize.volume(); i++) {
    //     match &= abs(in[i] - out[i]) < 0.00001;
    // }
    // if (match) {
    //     cout << "success" << endl;
    // } else {
    //     cout << "fail" << endl;
    // }
    return 0;
}
