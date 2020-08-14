#include "streamManager.hpp"

StreamManager::StreamManager() { CUDACHECK(cudaStreamCreate(&_stream)); }

StreamManager::~StreamManager() { CUDACHECK(cudaStreamDestroy(_stream)); }

void StreamManager::synchronize() { CUDACHECK(cudaStreamSynchronize(_stream)); }

cudaStream_t &StreamManager::stream() { return _stream; }