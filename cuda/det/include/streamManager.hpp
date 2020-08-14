#ifndef _STREAM_MANAGER_H
#define _STREAM_MANAGER_H

#include "commonCuda.hpp"

class StreamManager {
public:
  StreamManager();
  ~StreamManager();
  StreamManager(const StreamManager &) = delete;
  StreamManager &operator=(const StreamManager &) = delete;
  StreamManager(StreamManager &&) = delete;
  StreamManager &operator=(StreamManager &&) = delete;

  void synchronize();

  cudaStream_t &stream();

private:
  cudaStream_t _stream;
};

#endif