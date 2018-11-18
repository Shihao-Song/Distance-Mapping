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
void genFC(unsigned char *);
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
	unsigned char *raw_vol;

	raw_vol = (unsigned char *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(unsigned char));
	
	int i;
	for(i = 0; i < HEIGHT * WIDTH * DEPTH; i++)
	{
		raw_vol[i] = 0;
	}

	genFC(raw_vol); // Generate a feature cuboid 

	float sp[3] = {1.0, 2.5, 1.5}; // Voxel spacings, {i (height of a voxel), 
				//		j (width of a voxel), 
				//		k (depth of a voxel)}
	float sp2[3] = {
		sp[0] * sp[0],
	       	sp[1] * sp[1],
		sp[2] * sp[2]	
	};

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
	free(raw_vol);

	free(dist_mapping_ref);
	free(dist_mapping_maurer_openmp);

	return 1;
}

void genFC(unsigned char *vol)
{
	/* Distance between slices */
	int slice_stride = HEIGHT * WIDTH;

	int row_id, col_id, dep_id;

	/* Front and back (of the feature cuboid) */
	for (row_id = 1; row_id < (HEIGHT - 1); row_id++)
	{
		for (col_id = 1; col_id < (WIDTH - 1); col_id++)
		{
			// Front FVs should have negative K face
			vol[0 * slice_stride + row_id * WIDTH + col_id] |= 0x04;

			// Back FVs should have positive K face
			vol[(DEPTH - 1) * slice_stride + row_id * WIDTH + col_id] |= 0x20;
		}
	}

	/* Left and right */
	for (dep_id = 0; dep_id < DEPTH; dep_id++)
	{
		for (row_id = 1; row_id < (HEIGHT - 1); row_id++)
		{
			// Left FVs should have negative I face
			vol[dep_id * slice_stride + row_id * WIDTH + 1] |= 0x01;

			// Right FVs should have positive I face
			vol[dep_id * slice_stride + row_id * WIDTH + (WIDTH - 2)] |= 0x08;
		}
	}

	/* Top and bottom */
	for (dep_id = 0; dep_id < DEPTH; dep_id++)
	{
		for (col_id = 1; col_id < (WIDTH - 1); col_id++)
		{
			vol[dep_id * slice_stride + 1 * WIDTH + col_id] |= 0x02;
		
			vol[dep_id * slice_stride + (HEIGHT - 2) * WIDTH + col_id] |= 0x10;
		}
	}
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

