SOURCE	:= Run_Maurer.cpp ../src/maurer/maurer_distance_map.cxx

CC 	:= g++
FLAGS 	:= -O3 -fopenmp
TARGET	:= Run_Maurer


all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(FLAGS) -o $(TARGET) $(SOURCE)

clean:
	rm $(TARGET)
