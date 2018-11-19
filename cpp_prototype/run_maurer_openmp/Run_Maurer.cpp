#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>

#include "../src/maurer/maurer_distance_map.h"
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
	
	/***********************************************************
		Step two: intialize distance mapping outputs	
	************************************************************/
	double *dist_mapping_ref = (double *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(double));
	
	double *dist_mapping_maurer_openmp =
				(double *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(double));

	// Initialization
	for (int i = 0; i < HEIGHT * WIDTH * DEPTH; i++)
	{
		dist_mapping_ref[i] = -1.0;
		dist_mapping_maurer_openmp[i] = -1.0;
	}	

	/**************************************************************
		Step three: compute using exhaustive and maurer
	***************************************************************/
	
	
	/************************************************
		Step four: check the generated FT
	*************************************************/
	/*
	if (check(dist_mapping_ref, dist_mapping_maurer_openmp, HEIGHT * WIDTH * DEPTH) == 0)
	{
		printf("\nMaurer Testing: Error! \n");
	}
	else
	{
		printf("\nMaurer Testing: Successful! (Ref Solution: Exhaustive Search) \n");
	}
	*/

	/*
		Free memory resource
	*/

	free(dist_mapping_ref);
	free(dist_mapping_maurer_openmp);

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

