# CUDA_INSTALL_DIR = /home/seb/cuda-10.1/cuda-toolkit

CXX = g++-8
CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS = -I$(CUDA_INSTALL_DIR)/include -I./include
LDFLAGS = -L$(CUDA_INSTALL_DIR)/lib64 -L./lib
LDLIBS = -lcudart -lcnpy
NVCC = $(CUDA_INSTALL_DIR)/bin/nvcc
NVFLAGS = -g -Xcompiler -Wall

nmsCpu: main.o
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

nmsGpu: nmsGpu.obj
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

generateBoxes: generateBoxes.obj
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

loadNumpy: loadNumpy.o
	$(CXX) $(LDFLAGS) -lcnpy $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

%.obj: %.cu
	$(NVCC) $(NVFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj main nmsGpu nmsCpu loadNumpy generateBoxes
