#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
	Volume info
*/
#define HEIGHT 9
#define WIDTH 9
#define DEPTH 3

typedef unsigned char uchar;

/*
	Functions just for testing
*/
void genFV(uchar *); // Simply generate a feature cuboid as shown in report
void check(float *, float *); // Check the Maurer's solution with the reference solution

/*
	Debugging functions, do not invoke these if volume is huge.
*/
void printVolume(uchar *);

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		printf("Error: Please provide mapping scheme. ");
		printf("Example: %s --center-face \n", argv[0]);
		exit(0);
	}

	/*
		Step one: initialize testing volume
	*/
	uchar *raw_vol;

	raw_vol = (uchar *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(uchar));
	
	int i;
	for(i = 0; i < HEIGHT * WIDTH * DEPTH; i++)
	{
		raw_vol[i] = 0;
	}

	genFV(raw_vol); // Generate some feature voxels

	printVolume(raw_vol);
	/*
		Free memory resource
	*/
	free(raw_vol);

	return 1;
}

void genFV(uchar *vol)
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
