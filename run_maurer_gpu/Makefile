SOURCE		:= Run_Maurer.cu ../src/exhaustive_ft/exhaustiveFT.cpp ../src/maurer_openmp/maurer.cpp ../src/maurer_openmp/gnodes.cpp ../src/distance_transformation_openmp/distance_transformation.cpp ../src/distance_transformation_gpu/distance_transformation_gpu.cu
OBJ		:= Run_Maurer.o exhaustiveFT.o maurer.o gnodes.o distance_transformation.o distance_transformation_gpu.o
PROGRAM		:= Run_Maurer
NVCC		:= /usr/local/cuda-10.0/bin/nvcc
LD_LIBRARY_PATH	:= /usr/local/cuda-10.0/lib64/
FLAGS		:= -O3 -ccbin g++ -m64 --gpu-architecture=sm_61

all: $(PROGRAM)

$(PROGRAM): $(SOURCE)
	$(NVCC) $(FLAGS) --device-c $(SOURCE)
	$(NVCC) $(FLAGS) $(OBJ) --output-file $(PROGRAM) $(LIBS)

clean: 
	rm $(PROGRAM)
	rm *.o
