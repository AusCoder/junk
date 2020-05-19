CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS = -I$(CUDA_INSTALL_DIR)/include $(shell pkg-config --cflags opencv4) -I./include
LDFLAGS = -L$(CUDA_INSTALL_DIR)/lib64 -L./lib
LDLIBS = $(shell pkg-config --libs opencv4)  # -lcudart -lcnpy
NVCC = $(CUDA_INSTALL_DIR)/bin/nvcc
NVFLAGS = -g -Xcompiler -Wall

mainMtcnn: mainMtcnn.o mtcnn.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

nmsCpu: nmsCpu.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

nmsGpu: nmsGpu.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

generateBoxes: generateBoxes.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

cropResize: cropResize.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

loadNumpy: loadNumpy.o
	$(CXX) $^ -o $@ $(LDFLAGS) -lcnpy

loggerSample: loggerSample.o logger.o
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

%.obj: %.cu
	$(NVCC) $(NVFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj main nmsGpu nmsCpu generateBoxes cropResize loadNumpy mainMtcnn loggerSample
