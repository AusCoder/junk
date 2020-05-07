NVCC = /home/seb/cuda/cuda-10.1-src/cuda-toolkit/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

main: main.cpp
	$(NVCC) $(NVCC_FLAGS) $< -o $@
