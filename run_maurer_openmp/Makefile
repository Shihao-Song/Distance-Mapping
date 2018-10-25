SOURCE	:= Run_Maurer.cpp ../src/exhaustive_ft/exhaustiveFT.cpp ../src/maurer_openmp/maurer.cpp ../src/maurer_openmp/gnodes.cpp ../src/distance_transformation_openmp/distance_transformation.cpp

CC 	:= g++
FLAGS 	:= -O3 -fopenmp
TARGET	:= Run_Maurer


all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(FLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm $(TARGET)
