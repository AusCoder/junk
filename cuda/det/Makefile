CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS = -I$(CUDA_INSTALL_DIR)/include $(shell pkg-config --cflags opencv4) -I./include
LDFLAGS = -L$(CUDA_INSTALL_DIR)/lib64 -L./lib
LDLIBS = $(shell pkg-config --libs opencv4) -lnvinfer -lnvinfer_plugin -lnvparsers  -lcudart # -lcnpy
NVCC = $(CUDA_INSTALL_DIR)/bin/nvcc
NVFLAGS = -g -Xcompiler -Wall

mainMtcnn: mainMtcnn.o mtcnn.o trtNet.o logger.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

nmsCpu: nmsCpu.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

nmsGpu: nmsGpu.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

generateBoxes: generateBoxes.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

cropResize: cropResize.obj common.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

cropResizeCpu: cropResize.o common.o
	$(CXX) $^ -o $@ $(LDFLAGS) -lopencv_core -lopencv_imgcodecs

loadNumpy: loadNumpy.o
	$(CXX) $^ -o $@ $(LDFLAGS) -lcnpy

loggerSample: loggerSample.o logger.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

test: test_common.o common.o
	$(CXX) $^ -o $@ $(LDFLAGS) -lopencv_core -lopencv_imgcodecs

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

%.obj: %.cu
	$(NVCC) $(NVFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj main nmsGpu nmsCpu generateBoxes \
		cropResize cropResizeCpu loadNumpy mainMtcnn loggerSample test
