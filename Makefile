# -gencode:
# 	Parallel Thread Execution (PTX) is a pseudo-assembly 
#	language used in Nvidia's CUDA environment. The nvcc 
#	compiler translate code written in CUDA into PTX, and 
#	the graphics dirver contains a compiler which translate 
#	the PTX into a binary code

SOURCE				:= main.cu exhaustiveFT.cpp gnodes.cpp maurer.cpp
OBJ				:= main.o exhaustiveFT.o gnodes.o maurer.o
PROGRAM				:= main
NVCC				:= /usr/local/cuda-10.0/bin/nvcc
LD_LIBRARY_PATH			:= /usr/local/cuda-10.0/lib64/
FLAGS				:= -O3 -ccbin g++ -m64 --gpu-architecture=sm_61

all: $(PROGRAM)

main: $(SOURCE)
	$(NVCC) $(FLAGS) --device-c $(SOURCE)
	$(NVCC) $(FLAGS) $(OBJ) --output-file $(PROGRAM) $(LIBS)

clean: 
	rm $(PROGRAM)
	rm *.o

