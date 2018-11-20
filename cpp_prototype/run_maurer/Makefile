SOURCE	:= Run_Maurer.cu ../src/maurer/maurer_distance_map.cxx ../src/exhaustive/exhaustive_distance_map.cxx ../src/boundary_face_dist/boundary_face_distance.cxx

OBJ		:= Run_Maurer.o maurer_distance_map.o exhaustive_distance_map.o boundary_face_distance.o

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
