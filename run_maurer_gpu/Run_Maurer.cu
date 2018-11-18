#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "../src/maurer_openmp/maurer.h"
#include "../src/distance_transformation_openmp/distance_transformation.h" 

#include "../src/maurer_gpu/maurer_GPU.cuh"
#include "../src/distance_transformation_gpu/distance_transformation_gpu.cuh"

/*
	Volume info
*/
#define HEIGHT 250
#define WIDTH 250
#define DEPTH 125

/*
	Functions just for testing
*/
void genFC(unsigned char *); // Simply generate a feature cuboid
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
	
	// raw_vol will be transfered to GPU
	// cudaHostAlloc is similar to malloc but faster when transfering data to GPU
	if (cudaHostAlloc((void **)&raw_vol, HEIGHT * WIDTH * DEPTH * sizeof(unsigned char),
						cudaHostAllocDefault) != cudaSuccess)
	{
		printf("cudaHostAlloc() failed! \n");
		exit(0);
	}
	
	for(int i = 0; i < HEIGHT * WIDTH * DEPTH; i++)
	{
		raw_vol[i] = 0;
	}

	printf("\nGenerating testing data, volume size: %d x %d x %d \n", HEIGHT, WIDTH, DEPTH);
	genFC(raw_vol); // Generate a feature cuboid 

	float sp[3] = {1.0, 2.5, 1.5}; // Voxel spacings, {i (height of a voxel), 
				//		j (width of a voxel), 
				//		k (depth of a voxel)}

	float sp2[3] = {
		sp[0] * sp[0],
	       	sp[1] * sp[1],
		sp[2] * sp[2]	
	};

	/********************************************************************
		Step two: initialize distance mapping output for OpenMP	
	*********************************************************************/
	double *dist_mapping_maurer_openmp =
			(double *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(double));	
	
	// Initialization
	for (int i = 0; i < HEIGHT * WIDTH * DEPTH; i++)
	{
		dist_mapping_maurer_openmp[i] = -1.0;
	}	

	/*********************************************************************
		Step three: initialize distance mapping output for GPU 
	**********************************************************************/
	double *dist_mapping_maurer_gpu;
	if (cudaHostAlloc((void **)&dist_mapping_maurer_gpu,
				HEIGHT * WIDTH * DEPTH * sizeof(double),
				cudaHostAllocDefault) != cudaSuccess)
	{
		printf("cudaHostAlloc() failed! \n");
		exit(0);
	}
		
	/*******************************************************
		Step four: Allocate device (GPU) memory 
	********************************************************/
	// Memory location contains raw volume
	unsigned char *dev_raw_vol;
	if (cudaMalloc((void **)&dev_raw_vol, 
				HEIGHT * WIDTH * DEPTH * sizeof(unsigned char)) != cudaSuccess)
	{
		printf("cudaMalloc() failed.\n");
		exit(0);
	}

	// Transfer raw_vol to GPU
	if (cudaMemcpy(dev_raw_vol, raw_vol, HEIGHT * WIDTH * DEPTH * sizeof(unsigned char),
		cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("cudaMemcpy() failed! \n");
		exit(0);
	}

	// Memory location contains distance mapping result
	double *dev_dist_mapping_maurer_gpu[2]; // ping-pong technique

	for (int i = 0; i < 2; i++)
	{
		if (cudaMalloc((void **)&dev_dist_mapping_maurer_gpu[i],
				HEIGHT * WIDTH * DEPTH * sizeof(double)) != cudaSuccess)
        	{
                	printf("cudaMalloc() failed.\n");
                	exit(0);
        	}
	}
	
	/*****************************************************************************
		Step five: Distance mapping using OpenMP (Reference Solution)
	******************************************************************************/
	// Perform FT using maurer's OpenMP implementaion
	printf("\nPerforming Maurer's Distance Mapping using OpenMP...\n");
	struct timeval stopCPU, startCPU;
	gettimeofday(&startCPU, NULL);
	
	maurerFT(raw_vol, sp2,
		HEIGHT, WIDTH, DEPTH,
		dist_mapping_maurer_openmp);

	distTransformation(argv[1],
			raw_vol, sp2,
			HEIGHT, WIDTH, DEPTH,
			dist_mapping_maurer_openmp);

	gettimeofday(&stopCPU, NULL);
	long seconds = stopCPU.tv_sec - startCPU.tv_sec;
	long useconds = stopCPU.tv_usec - startCPU.tv_usec;
	long mtime = seconds * 1000 + useconds / 1000.0;
	printf("\nOpenMP Execution Time: %ld ms. \n", mtime);	

	/*******************************************************
		Step six: Distance mapping using GPU
	*******************************************************/
	// D1 Scan
	dim3 dimBlock_D1;
	dimBlock_D1.x = D1_NUM_THREADS_PER_BLOCK_X;
	dimBlock_D1.y = D1_NUM_THREADS_PER_BLOCK_Y;
	dimBlock_D1.z = D1_NUM_THREADS_PER_BLOCK_Z;
	
	dim3 dimGrid_D1;
	dimGrid_D1.x = 1;
	dimGrid_D1.y = int(ceil((double)HEIGHT / (double)dimBlock_D1.y));
	dimGrid_D1.z = int(ceil((double)DEPTH / (double)dimBlock_D1.z));

	// D2 Scan
	dim3 dimBlock_D2;
	dimBlock_D2.x = D2_NUM_THREADS_PER_BLOCK_X;
	dimBlock_D2.y = D2_NUM_THREADS_PER_BLOCK_Y;
	dimBlock_D2.z = D2_NUM_THREADS_PER_BLOCK_Z;
	
	dim3 dimGrid_D2;
	dimGrid_D2.x = int(ceil((double)WIDTH / (double)dimBlock_D2.x));
	dimGrid_D2.y = 1;
	dimGrid_D2.z = int(ceil((double)DEPTH / (double)dimBlock_D2.z));

	// D3 Scan
	dim3 dimBlock_D3;
	dimBlock_D3.x = D3_NUM_THREADS_PER_BLOCK_X;
	dimBlock_D3.y = D3_NUM_THREADS_PER_BLOCK_Y;
	dimBlock_D3.z = D3_NUM_THREADS_PER_BLOCK_Z;
	
	dim3 dimGrid_D3;
	dimGrid_D3.x = int(ceil((double)WIDTH / (double)dimBlock_D3.x));
	dimGrid_D3.y = int(ceil((double)HEIGHT / (double)dimBlock_D3.y));
	dimGrid_D3.z = 1;
	
	// For distance translation
	int scheme = 0;
	if (strcmp(argv[1], "--center-face") == 0)
	{
		scheme = 1;
	}
	else if (strcmp(argv[1], "--center-center") == 0)
	{
		scheme = 0;
	}
	
	dim3 dimBlock_DT;
	dimBlock_DT.x = NUM_THREADS_PER_BLOCK;
	dimBlock_DT.y = 1;
	dimBlock_DT.z = 1;

	dim3 dimGrid_DT;
       	dimGrid_DT.x = NUM_BLOCKS_PER_GRID;
        dimGrid_DT.y = 1;
        dimGrid_DT.z = 1;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	printf("\nPerforming Maurer's Distance Mapping using GPU...\n");
	cudaEventRecord(start);

	maurerFT_GPU<<<dimGrid_D1, dimBlock_D1>>>(dev_raw_vol, 0,
						HEIGHT, WIDTH, DEPTH,
						sp2[0], sp2[1], sp2[2],
						dev_dist_mapping_maurer_gpu[0]);
	
	maurerFT_GPU<<<dimGrid_D2, dimBlock_D2>>>(dev_dist_mapping_maurer_gpu[0], 1,
						HEIGHT, WIDTH, DEPTH,
                                                sp2[0], sp2[1], sp2[2],
                                                dev_dist_mapping_maurer_gpu[1]);
	
	maurerFT_GPU<<<dimGrid_D3, dimBlock_D3>>>(dev_dist_mapping_maurer_gpu[1], 2,
						HEIGHT, WIDTH, DEPTH,
                                                sp2[0], sp2[1], sp2[2],
            					dev_dist_mapping_maurer_gpu[0]);

	distTransformation_GPU<<<dimGrid_DT, dimBlock_DT>>>(scheme,
							dev_raw_vol,
                        				sp2[0], sp2[1], sp2[2],
							HEIGHT, WIDTH, DEPTH,
							dev_dist_mapping_maurer_gpu[0]);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("\nExecution Time on GPU: %f ms\n", milliseconds);
	/*************************************************************
		Step seven: transfer the results back to CPU 
	**************************************************************/
	if (cudaMemcpy(dist_mapping_maurer_gpu, dev_dist_mapping_maurer_gpu[0],
			HEIGHT * WIDTH * DEPTH * sizeof(double),
			cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("cudaMemcpy() failed! \n");
		exit(0);
	}

	/***********************************
		Step eight: checking 
	************************************/
	if (check(dist_mapping_maurer_openmp,
		dist_mapping_maurer_gpu,
		HEIGHT * WIDTH * DEPTH) == 0)
        {
                printf("\nMaurer GPU Testing: Error! \n");
        }
        else
        {
                printf("\nMaurer GPU Testing: Successful! (Ref Solution: OpenMP) \n");
        }

	/*
		Free memory resource
	*/
	cudaFreeHost(raw_vol);

	free(dist_mapping_maurer_openmp);

	cudaFreeHost(dist_mapping_maurer_gpu);

	cudaFree(dev_raw_vol);
	for (int i = 0; i < 2; i++)
	{
		cudaFree(dev_dist_mapping_maurer_gpu[i]);
	}

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

int check(double *ref, double *output, int length)
{
	int i;
	for(i = 0; i < length; i++)
	{
		if(ref[i] != output[i])
		{
			return 0;
		}
	}

	return 1;
}
