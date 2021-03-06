CXXFLAGS = -g -Wall -std=c++14
CPPFLAGS = -I./include -I/opt/cnpy/include -I/opt/cub -I$(CUDA_INSTALL_DIR)/include $(shell pkg-config --cflags opencv4 jsoncpp)
LDFLAGS = -L./lib -L/opt/cnpy/lib -L$(CUDA_INSTALL_DIR)/lib64
LDLIBS = $(shell pkg-config --libs opencv4 jsoncpp) -lnvinfer -lnvinfer_plugin -lnvparsers -lcudart -lcnpy
NVCC = $(CUDA_INSTALL_DIR)/bin/nvcc
NVFLAGS = -g -G -Xcompiler -Wall -Xcompiler -std=c++14

mainMtcnn: mainMtcnn.o mtcnn.o trtNet.o trtNetInfo.o commonCuda.o streamManager.o \
					logger.o mtcnnKernels.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

nmsCpu: nmsCpu.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

nmsGpu: nmsGpu.obj mtcnnKernels.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

generateBoxes: generateBoxes.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

cropResizeGpu: cropResizeGpu.o common.o mtcnnKernels.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

cropResizeCpu: cropResizeCpu.o common.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

loadNumpy: loadNumpy.o
	$(CXX) $^ -o $@ $(LDFLAGS) -lcnpy

loggerSample: loggerSample.o logger.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

trtNetMain: trtNetMain.o trtNet.o trtNetInfo.o commonCuda.o logger.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

nicerDeviceApi: nicerDeviceApi.o streamManager.o mtcnnKernels.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

test_common: test_common.o common.o
	$(CXX) $^ -o $@ $(LDFLAGS) -lopencv_core -lopencv_imgcodecs

test_trtNet: test_trtNet.o trtNet.o trtNetInfo.o commonCuda.o logger.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

test_mtcnnKernels: test_mtcnnKernels.o commonCuda.o streamManager.o mtcnnKernels.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

test_tests: test_tests.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

moderngpuHello:
	nvcc -std=c++11 --expt-extended-lambda -gencode arch=compute_61,code=compute_61 \
		-I /opt/src/moderngpu/src -o moderngpuHello moderngpuHello.cu

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

%.obj: %.cu
	$(NVCC) $(NVFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj main nmsGpu nmsCpu generateBoxes \
		cropResizeGpu cropResizeCpu loadNumpy mainMtcnn loggerSample test_common \
		test_trtNet test_mtcnnKernels trtNetMain moderngpuHello nicerDeviceApi
