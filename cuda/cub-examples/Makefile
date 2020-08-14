CXXFLAGS = -g -Wall -std=c++14
CPPFLAGS = -I/opt/cub -I$(CUDA_INSTALL_DIR)/include
LDFLAGS = -L$(CUDA_INSTALL_DIR)/lib64
LDLIBS = -lcudart
NVCC = $(CUDA_INSTALL_DIR)/bin/nvcc
NVFLAGS = -g -G -Xcompiler -Wall -Xcompiler -std=c++14

reduce: reduce.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

sort: sort.obj
	$(CXX) $^ -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c $< -o $@

%.obj: %.cu
	$(NVCC) $(NVFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	$(RM) *.o *.obj reduce sort
