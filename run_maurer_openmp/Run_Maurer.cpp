#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "../src/exhaustive_ft/exhaustiveFT.h"
#include "../src/maurer_openmp/maurer.h"
#include "../src/distance_transformation_openmp/distance_transformation.h" 

/*
	Volume info

	The exhaustive search (reference solution) will take significant amount of time if
	the volume is huge!
*/
#define HEIGHT 7
#define WIDTH 7
#define DEPTH 10

typedef unsigned char uchar;

/*
	Functions just for testing
*/
void genFC(uchar *); // Simply generate a feature cuboid as shown in report (has boundary face)
void genRandomFV(uchar *, int); // Generate random number of feature voxels (no boundary face)
int check(int *, int *, int); // Check the Maurer's FT with the reference FT

/*
	Debugging functions, do not invoke these if volume is huge.
*/
void printVolume(uchar *);
void printDistTransformation(float *);

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
	uchar *raw_vol;

	raw_vol = (uchar *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(uchar));
	
	int i;
	for(i = 0; i < HEIGHT * WIDTH * DEPTH; i++)
	{
		raw_vol[i] = 0;
	}

	genFC(raw_vol); // Generate a feature cuboid 
	//genRandomFV(raw_vol, 60);

	float sp[3] = {1.0, 1.0, 1.0}; // Voxel spacings, {i (height of a voxel), 
				//		j (width of a voxel), 
				//		k (depth of a voxel)}
	float sp2[3] = {
		sp[0] * sp[0],
	       	sp[1] * sp[1],
		sp[2] * sp[2]	
	};

	/*********************************************
		Step two: initialize FT output	
	**********************************************/
	// ref_ft : the FT performed by exhaustive search (src/exhaustive_ft)
	int *ref_ft = (int *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(int));
	
	// maurer_ft : the FT performed by MAURER
	int *maurer_ft = (int *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(int));

	// Initialization
	for (int i = 0; i < HEIGHT * WIDTH * DEPTH; i++)
	{
		ref_ft[i] = -1;
		maurer_ft[i] = -1;
	}	

	/**************************************************************
		Step three: compute using exhaustive and maurer
	***************************************************************/
	// Perform FT using exhaustive search
	exhaustiveFT(raw_vol, sp2, HEIGHT, WIDTH, DEPTH, ref_ft);

	// Perform FT using maurer's
	struct timeval stopCPU, startCPU;
	gettimeofday(&startCPU, NULL);
	maurerFT(raw_vol, sp2, HEIGHT, WIDTH, DEPTH, maurer_ft);
	gettimeofday(&stopCPU, NULL);
	long seconds = stopCPU.tv_sec - startCPU.tv_sec;
	long useconds = stopCPU.tv_usec - startCPU.tv_usec;
	long mtime = seconds * 1000 + useconds / 1000.0;
	printf("\nOpenMP Execution Time of Maurer FT: %ld ms. \n", mtime);	

	/************************************************
		Step four: check the generated FT
	*************************************************/
	if (check(ref_ft, maurer_ft, HEIGHT * WIDTH * DEPTH) == 0)
	{
		printf("\nMaurer Testing: Error! \n");
	}
	else
	{
		printf("\nMaurer Testing: Successful! (Ref Solution: Exhaustive Search) \n");
	}

	/*************************************************************
		Step five: perform distance mapping for maurer FT
	**************************************************************/
	printf("\nPerforming final distance transformation for maurer's FT. \n");
		
	float *dist_trans = (float *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(float));	

	gettimeofday(&startCPU, NULL);
	distTransformation(argv[1], maurer_ft, raw_vol, sp2, HEIGHT, WIDTH, DEPTH, dist_trans);
	gettimeofday(&stopCPU, NULL);
	seconds = stopCPU.tv_sec - startCPU.tv_sec;
	useconds = stopCPU.tv_usec - startCPU.tv_usec;
	mtime = seconds * 1000 + useconds / 1000.0;
	printf("\nOpenMP Execution Time of Distance Transformation: %ld ms. \n", mtime);

	printDistTransformation(dist_trans);
	
	/*
		Free memory resource
	*/
	free(raw_vol);

	free(ref_ft);
	free(maurer_ft);
	free(dist_trans);

	return 1;
}

void genFC(uchar *vol)
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

// percent% of the voxels will be set to feature
void genRandomFV(uchar *vol, int percent)
{
	int num_fv = HEIGHT * WIDTH * DEPTH * percent / 100;

	// Random seed
	srand(time(NULL));

	// Distance between slices
	int slice_stride = HEIGHT * WIDTH;

	int ite;
	for (ite = 0; ite < num_fv; ite++)
	{
		int r_row = rand() % HEIGHT;
		int r_col = rand() % WIDTH;
		int r_dep = rand() % DEPTH;

		vol[r_dep * slice_stride + r_row * WIDTH + r_col] = 1;
	}	
}

void printVolume(uchar *vol)
{
	// Distance between slices
	int slice_stride = HEIGHT * WIDTH;

	int i, j, k;

	for (k = 0; k < DEPTH; k++)
	{
		printf("Image Slice: %d\n", k);

		for (i = 0; i < HEIGHT; i++)
		{
			for (j = 0; j < WIDTH; j++)
			{
				printf("0x%02x ", vol[k * slice_stride + i * WIDTH + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

int check(int *ref, int *result, int length)
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

void printDistTransformation(float *dist_trans)
{
	int slice_stride = HEIGHT * WIDTH;

        int i, j, k;

        for (k = 0; k < DEPTH; k++)
        {
                printf("Image Slice: %d\n", k);

                for (i = 0; i < HEIGHT; i++)
                {
                        for (j = 0; j < WIDTH; j++)
                        {
                                printf("%4f  ", dist_trans[k * slice_stride + i * WIDTH + j]);
                        }
                        printf("\n");
                }
                printf("\n");
        }
}
