CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS = -I$(CUDA_INSTALL_DIR)/include -I$(TRT_INCLUDE_DIR)
LDFLAGS = -L$(CUDA_INSTALL_DIR)/lib64
LDLIBS = -lnvinfer
NVCC = $(CUDA_INSTALL_DIR)/bin/nvcc
NVFLAGS = -g -Xcompiler -Wall

loggerSample: loggerSample.o logger.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

%.obj: %.cu
	$(NVCC) $(NVFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj loggerSample
