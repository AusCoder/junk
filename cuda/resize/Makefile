CUDA_INSTALL_DIR = /mnt/data/opt/cuda-10.1/cuda-toolkit

CXX = g++-8
CXXFLAGS = -g -Wall -std=c++17
CPPFLAGS = -I$(CUDA_INSTALL_DIR)/include
LDFLAGS = -L$(CUDA_INSTALL_DIR)/lib64
LDLIBS = -lcudart
NVCC = $(CUDA_INSTALL_DIR)/bin/nvcc
NVFLAGS = -g -Xcompiler -Wall

main: main.o resize.obj
	$(CXX) $(LDFLAGS) $(LDLIBS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

%.obj: %.cu
	$(NVCC) $(NVFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) main *.o *.obj
