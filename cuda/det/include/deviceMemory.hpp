#ifndef _DEVICE_MEMORY_H
#define _DEVICE_MEMORY_H

#include "commonCuda.hpp"
#include <stdexcept>
#include <utility>
#include <vector>

// TODO: add a ConstDevicePtr<T>?
template <typename T> class DevicePtr {
public:
  __host__ __device__ __inline__ explicit DevicePtr(T *p,
                                                    std::size_t numElements)
      : _p{p}, _sizeBytes{sizeof(T) * numElements} {}

  __host__ __device__ __inline__ T *get() { return _p; }
  // A const pointer if we are const
  __host__ __device__ __inline__ const T *get() const { return _p; }

  // Return a reference that can be modified if we are non const
  __host__ __device__ __inline__ T &operator[](std::size_t idx) {
    return _p[idx];
  }
  // Return a value if we are const
  __host__ __device__ __inline__ T operator[](std::size_t idx) const {
    return _p[idx];
  }

  __host__ __device__ __inline__ std::size_t size() const {
    return _sizeBytes / sizeof(T);
  }
  __host__ __device__ __inline__ std::size_t sizeBytes() const {
    return _sizeBytes;
  }

  std::vector<T> asVec(cudaStream_t &stream) const {
    std::vector<T> output(size());
    CopyAllElementsAsync(output, *this, stream);
    return output;
  }

private:
  T *_p = nullptr;
  std::size_t _sizeBytes = 0;
};

template <typename T>
__host__ inline auto MakeDevicePtr(T *p, std::size_t numElements) {
  return DevicePtr<T>(p, numElements);
}

template <typename T> class DeviceMemory {
public:
  static DeviceMemory<T> AllocateElements(std::size_t numElements) {
    return DeviceMemory<T>(sizeof(T) * numElements);
  }
  static DeviceMemory<T> AllocateBytes(std::size_t numBytes) {
    return DeviceMemory<T>(numBytes);
  }

  DeviceMemory() = default;
  DeviceMemory(const DeviceMemory<T> &) = delete;
  DeviceMemory &operator=(const DeviceMemory<T> &) = delete;

  DeviceMemory(DeviceMemory<T> &&other) {
    std::swap(_p, other._p);
    std::swap(_sizeBytes, other._sizeBytes);
  };
  DeviceMemory &operator=(DeviceMemory<T> &&other) {
    std::swap(_p, other._p);
    std::swap(_sizeBytes, other._sizeBytes);
  };

  ~DeviceMemory() {
    if (_p) {
      CUDACHECK(cudaFree(_p));
    }
  }

  // Some useful methods to access underlying pointer
  T *get() { return _p; }
  const T *get() const { return _p; }

  // A conversion operator for convenience
  // Adding the explicit here is basically forcing us to use
  // static_cast<DevicePtr<T>> when we want to pass it to a function.
  //
  //  explicit operator DevicePtr<T>() const { return DevicePtr<T>(_p, size());
  //  }
  operator DevicePtr<T>() const { return DevicePtr<T>(_p, size()); }

  std::size_t size() const { return _sizeBytes / sizeof(T); }
  std::size_t sizeBytes() const { return _sizeBytes; }

  std::vector<T> asVec(cudaStream_t &stream) const {
    std::vector<T> output(size());
    CopyAllElementsAsync(output, *this, stream);
    return output;
  }

private:
  T *_p = nullptr;
  std::size_t _sizeBytes = 0;

  DeviceMemory(std::size_t sizeBytes) : _sizeBytes{sizeBytes} {
    CUDACHECK(cudaMalloc(&_p, sizeBytes));
  }
};

template <typename T>
void CopyElementsAsync(DevicePtr<T> dst, const T *src, std::size_t numElements,
                       cudaStream_t &stream) {
  if (dst.size() < numElements) {
    throw std::invalid_argument("dst DevicePtr not large enough");
  }
  CUDACHECK(cudaMemcpyAsync(
      static_cast<void *>(dst.get()), static_cast<const void *>(src),
      sizeof(T) * numElements, cudaMemcpyHostToDevice, stream));
}

template <typename T>
void CopyElementsAsync(DeviceMemory<T> &dst, const T *src,
                       std::size_t numElements, cudaStream_t &stream) {
  CopyElementsAsync(static_cast<DevicePtr<T>>(dst), src, numElements, stream);
}

template <typename T>
void SetElementAsync(DevicePtr<T> &dst, T value, cudaStream_t &stream) {
  CopyElementsAsync(dst, &value, 1, stream);
}

template <typename T>
void SetElementAsync(DeviceMemory<T> &dst, T value, cudaStream_t &stream) {
  CopyElementsAsync(static_cast<DevicePtr<T>>(dst), &value, 1, stream);
}

/* Assumes that src has exactly dst.size() elements
 */
template <typename T>
void CopyAllElementsAsync(DevicePtr<T> dst, const T *src,
                          cudaStream_t &stream) {
  CopyElementsAsync(dst, src, dst.size(), stream);
}

template <typename T>
void CopyAllElementsAsync(DeviceMemory<T> &dst, const T *src,
                          cudaStream_t &stream) {
  CopyElementsAsync(static_cast<DevicePtr<T>>(dst), src, dst.size(), stream);
}

/* Potentially confusing that this uses src.size() instead of dst.size()
 */
template <typename T>
void CopyAllElementsAsync(DevicePtr<T> dst, const std::vector<T> &src,
                          cudaStream_t &stream) {
  CopyElementsAsync(dst, src.data(), src.size(), stream);
}

template <typename T>
void CopyAllElementsAsync(DeviceMemory<T> &dst, const std::vector<T> &src,
                          cudaStream_t &stream) {
  CopyElementsAsync(static_cast<DevicePtr<T>>(dst), src.data(), src.size(),
                    stream);
}

template <typename T>
void CopyElementsAsync(T *dst, DevicePtr<T> src, std::size_t numElements,
                       cudaStream_t &stream) {
  if (src.size() < numElements) {
    throw std::invalid_argument("src DevicePtr not large enough");
  }
  CUDACHECK(cudaMemcpyAsync(
      static_cast<void *>(dst), static_cast<const void *>(src.get()),
      sizeof(T) * numElements, cudaMemcpyDeviceToHost, stream));
}

template <typename T>
void CopyElementsAsync(T *dst, const DeviceMemory<T> &src,
                       std::size_t numElements, cudaStream_t &stream) {
  CopyElementsAsync(dst, static_cast<DevicePtr<T>>(src), numElements, stream);
}

/* Assumes that dst can store src.size() elements
 */
template <typename T>
void CopyAllElementsAsync(T *dst, DevicePtr<T> src, cudaStream_t &stream) {
  CopyElementsAsync(dst, src, src.size(), stream);
}

template <typename T>
void CopyAllElementsAsync(T *dst, const DeviceMemory<T> &src,
                          cudaStream_t &stream) {
  CopyAllElementsAsync(dst, static_cast<DevicePtr<T>>(src), stream);
}

template <typename T>
void CopyAllElementsAsync(std::vector<T> &dst, DevicePtr<T> src,
                          cudaStream_t &stream) {
  if (dst.size() < src.size()) {
    throw std::invalid_argument("dst vector not large enough");
  }
  CopyElementsAsync(dst.data(), src, src.size(), stream);
}

template <typename T>
void CopyAllElementsAsync(std::vector<T> &dst, const DeviceMemory<T> &src,
                          cudaStream_t &stream) {
  CopyAllElementsAsync(dst, static_cast<DevicePtr<T>>(src), stream);
}

#endif