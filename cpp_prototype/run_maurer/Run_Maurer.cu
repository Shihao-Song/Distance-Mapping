#include <cstdio>
#include <cstdlib>

#include "../src/exhaustive/exhaustive_distance_map.h"
#include "../src/maurer/maurer_distance_map.h"
#include "../src/maurer_gpu/maurer_distance_map_gpu.cuh"
#include "../src/tmp/Vol.h"

#define HEIGHT 64
#define WIDTH 64
#define DEPTH 16

/*
	Functions just for testing
*/
int check(double *, double *, int);

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("Error: Please provide mapping scheme. ");
		printf("Example: %s --center-face \n", argv[0]);
		exit(0);
	}

	/***************************************************
		Step one: initialize testing volume
	****************************************************/
	Vol vol(HEIGHT, WIDTH, DEPTH);	

	/**************************************************************
		Step two: compute using exhaustive and maurer
	***************************************************************/
	Maurer_Distance_Map maurer(argv[1]);
	maurer.run_maurer(&vol);

	Exhaustive_Distance_Map exhaustive(argv[1]);
	exhaustive.run_exhaustive(&vol);

	Maurer_Distance_Map_GPU maurer_gpu(argv[1]);
	maurer_gpu.run_maurer_gpu(&vol);

	/************************************************
		Step four: check the generated FT
	*************************************************/
	if (check(exhaustive.dist_mapping_exhaustive, 
			maurer.dist_mapping_maurer_openmp, 
			HEIGHT * WIDTH * DEPTH) == 0)
	{
		printf("\nmaurer openmp: error! \n");
	}
	else
	{
		printf("\nmaurer openmp: successful! (ref solution: exhaustive search) \n");
	}
	
	if (check(exhaustive.dist_mapping_exhaustive, 
			maurer_gpu.dist_mapping_maurer_gpu, 
			HEIGHT * WIDTH * DEPTH) == 0)
	{
		printf("\nmaurer GPU: error! \n");
	}
	else
	{
		printf("\nmaurer GPU: successful! (ref solution: exhaustive search) \n");
	}

	return 1;
}

int check(double *ref, double *result, int length)
{
	int i;
	for(i = 0; i < length; i++)
	{
		if(ref[i] != result[i])
		{
			return 0;
		}
	}

	return 1;
}

