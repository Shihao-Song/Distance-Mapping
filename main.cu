#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#include "volume.h"
#include "exhaustiveFT.h"
#include "maurer.h"
#include "maurer_GPU.cuh"

/*
	Testing Functions are Defined here
*/
void initVolume(uchar *, int, int, int);
void printVolume(uchar *, int, int, int);
int check(int *, int *, int);

void checkResult(uchar *input, int *vol, int height, int width, int depth);
double calcDist(double i, double j, double target_i, double target_j);

bool isKthBitSet(int n, int k) 
{ 
    if (n & (1 << (k - 1))) 
        return 1; 
    else
        return 0; 
}

int main()
{
	// Initialize Input and Output Data
	/*	ref will always stay in host */
	int *ref = (int *)malloc(HEIGHT * WIDTH * DEPTH * sizeof(int));
	if (ref == NULL)
	{
		printf("malloc() failed! \n");
		return 0;
	}

	/* Input will be transfered to GPU */
	uchar *input;
	if (cudaHostAlloc((void **)&input, HEIGHT * WIDTH * DEPTH * sizeof(uchar), \
						cudaHostAllocDefault) != cudaSuccess)
	{
		printf("cudaHostAlloc() failed! \n");
		return 0;
	}

	/* Output will be transfered back to CPU */
	int *output_0;
	if (cudaHostAlloc((void **)&output_0, HEIGHT * WIDTH * DEPTH * sizeof(int), \
						cudaHostAllocDefault) != cudaSuccess)
	{
		printf("cudaHostAlloc() failed! \n");
		return 0;
	}

	int *output_1;
	if (cudaHostAlloc((void **)&output_1, HEIGHT * WIDTH * DEPTH * sizeof(int), \
						cudaHostAllocDefault) != cudaSuccess)
	{
		printf("cudaHostAlloc() failed! \n");
		return 0;
	}

	for (int i = 0; i < HEIGHT * WIDTH * DEPTH; i++)
	{
		output_0[i] = -1;
		output_1[i] = -1;
		ref[i] = -1;
	}

	// Generate Testing Data
	initVolume(input, HEIGHT, WIDTH, DEPTH);	

	// Print Volume
	printVolume(input, HEIGHT, WIDTH, DEPTH);

	// Generate Reference Result
	exhaustiveFT(input, HEIGHT, WIDTH, DEPTH, ref);

	

	// Compute using Maurer
	struct timeval stopCPU, startCPU;
	gettimeofday(&startCPU, NULL);
	maurerFT(input, HEIGHT, WIDTH, DEPTH, output_0);
	gettimeofday(&stopCPU, NULL);
	long seconds = stopCPU.tv_sec - startCPU.tv_sec;
	long useconds = stopCPU.tv_usec - startCPU.tv_usec;
	long mtime = seconds * 1000 + useconds / 1000.0;
	//printf("CPU Execution Time: %ld ms. \n", mtime);

	// Check the closet boundary point
		

	checkResult(input, output_0, HEIGHT, WIDTH, DEPTH);

	/*
		GPU Solutions
	*/
	/* CUDA Performance */
	/*
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);


	// Allocate device memory for input data
	uchar *dev_vol;
	if (cudaMalloc((void **)&dev_vol, HEIGHT * WIDTH * DEPTH * sizeof(uchar)\
		) != cudaSuccess)
	{
		printf("cudaMalloc() failed.\n");
		return 0;
	}
	
	// Allocate device memory for ping-pong I/O
	int *dev_ping_pong_buf[2];

	for (int i = 0; i < 2; i++)
	{
		if (cudaMalloc((void **)&(dev_ping_pong_buf[i]), HEIGHT * WIDTH * DEPTH * sizeof(int)\
			) != cudaSuccess)
		{
			printf("cudaMalloc() failed! \n");
			return 0;
		}	
	}

	// Bind texture memory - Remember to Unbind!
	// if (cudaBindTexture( NULL, ping_pong_tex_0, dev_ping_pong_buf[0], HEIGHT * WIDTH * DEPTH * sizeof(int)) \
		!= cudaSuccess)
	// {
	//	printf("cudaBindTexture() failed! \n");
	//	return 0;
	// }

	// Copy input data to GPU
	cudaMemcpy(dev_vol, input, HEIGHT * WIDTH * DEPTH * sizeof(uchar), \
		cudaMemcpyHostToDevice);

	// Computation
	dim3 dimBlock_D1_V1(256, 2, 2);
	dim3 dimGrid_D1_V1(1, 128, 32);

	dim3 dimBlock_D2_V1(2, 256, 2);
	dim3 dimGrid_D2_V1(128, 1, 32);

	//dim3 dimBlock_D3(4, 4, 64);
	//dim3 dimGrid_D3(64, 64, 1);

	dim3 dimBlock_D1_V2(32, 8, 4);
	dim3 dimGrid_D1_V2(1, 4, 2);
	//dim3 dimGrid_D1_V2(1, 32, 16);

	dim3 dimBlock_D2_V2(8, 32, 4);
	dim3 dimGrid_D2_V2(4, 1, 2);
	//dim3 dimGrid_D2_V2(32, 1, 16);

	//dim3 dimBlock_D3_V2(8, 4, 32);
	//dim3 dimGrid_D3_V2(32, 64, 1);
	dim3 dimBlock_D3_V2(8, 4, 8);
	dim3 dimGrid_D3_V2(4, 8, 1);


	cudaEventRecord(start);
	//initVol_GPU<<<360, 256>>>(dev_vol, HEIGHT, WIDTH, DEPTH, dev_output);
	//raster_scan_GPU_v1<<<dimGrid_D1_V1, dimBlock_D1_V1>>>(dev_vol, 1, WIDTH, HEIGHT, WIDTH, DEPTH, \
															dev_ping_pong_buf[0]);
	raster_scan_GPU_v2<<<dimGrid_D1_V2, dimBlock_D1_V2>>>(dev_vol, 0, HEIGHT, WIDTH, DEPTH, \
															dev_ping_pong_buf[1]);
	raster_scan_GPU_v2<<<dimGrid_D2_V2, dimBlock_D2_V2>>>(dev_ping_pong_buf[1], 1, HEIGHT, WIDTH, DEPTH, \
															dev_ping_pong_buf[0]);
	raster_scan_GPU_v2<<<dimGrid_D3_V2, dimBlock_D3_V2>>>(dev_ping_pong_buf[0], 2, HEIGHT, WIDTH, DEPTH, \
															dev_ping_pong_buf[1]);

	//raster_scan_GPU_v2<<<dimGrid_D3_V2, dimBlock_D3_V2>>>(dev_ping_pong_buf[0], 2, HEIGHT, WIDTH, DEPTH, \
															dev_ping_pong_buf[1]);
	//raster_scan_GPU_v1<<<dimGrid_D2_V1, dimBlock_D2_V1>>>(dev_ping_pong_buf[0], 2, HEIGHT, HEIGHT, WIDTH, DEPTH, \
												dev_ping_pong_buf[1]);
	// Different buffer is needed since the original data may get overwritten
	//raster_scan_GPU_v2<<<dimGrid_D2_V2, dimBlock_D2_V2>>>(dev_ping_pong_buf[0], 1, HEIGHT, WIDTH, DEPTH, \
															dev_ping_pong_buf[1]);

	// maurer_GPU<<<dimGrid_D3, dimBlock_D3>>>(dev_vol, 3, DEPTH, HEIGHT, WIDTH, DEPTH, dev_output);
	
	// Send result back to host
	//cudaMemcpy(output_0, dev_ping_pong_buf[0], HEIGHT * WIDTH * DEPTH * sizeof(int), \
		cudaMemcpyDeviceToHost);
	cudaMemcpy(output_1, dev_ping_pong_buf[1], HEIGHT * WIDTH * DEPTH * sizeof(int), \
		cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Execution Time on GPU: %f ms\n", milliseconds);

	// Free device resource
	// cudaUnbindTexture(ping_pong_tex_0);

	cudaFree(dev_vol);
	for (int i = 0; i < 2; i++)
	{
		cudaFree(dev_ping_pong_buf[i]);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	*/

	// Check result
	if (check(ref, output_0, HEIGHT * WIDTH * DEPTH) == 0)
	{
		printf("Error! \n");
	}
	else
	{
		printf("Successful! \n");
	}

	/*
	if (check(ref, output_1, HEIGHT * WIDTH * DEPTH) == 0)
	{
		printf("GPU V2 Error! \n");
	}
	else
	{
		printf("GPU V2 Successful! \n");
	}
	*/
	
	// De-allocate Memory
	cudaFreeHost(input);
	cudaFreeHost(output_0);
	cudaFreeHost(output_1);
	free(ref);	
}

void initVolume(uchar *vol, int height, int width, int depth)
{
	for (int i = 0; i < HEIGHT * WIDTH * DEPTH; i++)
	{
		 vol[i] = 0;
	}

	srand(time(NULL));

	// Distance between slices
	int slice_stride = height * width;

	/*
	for (int ite = 0; ite < 5; ite++)
	{
		int r_row = rand() % height;
		int r_col = rand() % width;
		int r_dep = rand() % depth;

		vol[r_dep * slice_stride + r_row * width + r_col] = 1;
	}
	*/
	
	int r_row = 3;
	int r_col = 3;
	int r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x03;

	r_row = 3;
	r_col = 4;
	r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x02;

	r_row = 3;
	r_col = 5;
	r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x0A;

	r_row = 4;
	r_col = 3;
	r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x01;

	r_row = 4;
	r_col = 6;
	r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x0A;

	r_row = 5;
	r_col = 3;
	r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x11;

	r_row = 5;
	r_col = 6;
	r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x08;

	r_row = 6;
	r_col = 4;
	r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x11;
	
	r_row = 6;
	r_col = 5;
	r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x10;

	r_row = 6;
	r_col = 6;
	r_dep = 0;
	vol[r_dep * slice_stride + r_row * width + r_col] = 0x18;
}

void printVolume(uchar *vol, int height, int width, int depth)
{
	// Distance between slices
	int slice_stride = height * width;

	int i, j, k;

	for (k = 0; k < depth; k++)
	{
		printf("Image Slice: %d\n", k);

		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				printf("0x%02x ", vol[k * slice_stride + i * width + j]);
			}
			printf("\n");
		}
		printf("\n");
	}
}

void checkResult(uchar *input, int *vol, int height, int width, int depth)
{
	// Distance between slices
	int slice_stride = height * width;

	int i, j, k;

	for (k = 0; k < depth; k++)
	{
		printf("Image Slice: %d\n", k);

		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				int row_id = vol[k * slice_stride + i * width + j] / width;
				int col_id = vol[k * slice_stride + i * width + j] % width;
				
				if (row_id == i && col_id == j)
				{
					printf("**********  ");
				}
				else
				{
					int input_idx = vol[k * slice_stride + i * width + j];
					//printf("(%d, %d) \n", row_id, col_id);
					//printf("Value: %d \n",input[input_idx]);
					double temp_dist = 1000000.0;
					
					int face_id = 0;
					int point_id = 0;
					
					// If has negative i boundary
					if (isKthBitSet(int(input[input_idx]), 1))
					{
					//	printf("Has negative i boundary. \n");
						double dist_0 = calcDist(double(i), \
									double(j), \
									double(row_id + 0.5), \
									double(col_id - 0.5));
						//printf("%f\n", dist_0);

						double dist_1 = calcDist(double(i), \
									double(j), \
									double(row_id), \
									double(col_id - 0.5));
						
						//printf("%f\n", dist_1);
						double dist_2 = calcDist(double(i), \
									double(j), \
									double(row_id - 0.5), \
									double(col_id - 0.5));
			
						//printf("%f\n", dist_2);
						if (dist_0 < temp_dist)
						{
							temp_dist = dist_0;
							face_id = 0;
							point_id = 0;
						}
						if (dist_1 < temp_dist)
						{
							temp_dist = dist_1;
							face_id = 0;
							point_id = 1;
						}
						
						if (dist_2 < temp_dist)
						{
							temp_dist = dist_2;
							face_id = 0;
							point_id = 2;
						}
					}

					// If has positive i boundary
					if (isKthBitSet(int(input[input_idx]), 4))
					{
					//	printf("Check: %d\n", input[input_idx] & 0x08);
					//	printf("Has positive i boundary. \n");
						double dist_0 = calcDist(double(i), \
									double(j), \
									double(row_id - 0.5), \
									double(col_id + 0.5));
						
						double dist_1 = calcDist(double(i), \
									double(j), \
									double(row_id), \
									double(col_id + 0.5));
						
						double dist_2 = calcDist(double(i), \
									double(j), \
									double(row_id + 0.5), \
									double(col_id + 0.5));
			
						if (dist_0 < temp_dist)
						{
							temp_dist = dist_0;
							face_id = 2;
							point_id = 0;

						}
						if (dist_1 < temp_dist)
						{
							temp_dist = dist_1;
							face_id = 2;
							point_id = 1;
						}
						
						if (dist_2 < temp_dist)
						{
							temp_dist = dist_2;
							face_id = 2;
							point_id = 2;
						}
					}
					
					// If has negative j boundary
					if (isKthBitSet(int(input[input_idx]), 2))
					{
					//	printf("Has negative j boundary. \n");
						double dist_0 = calcDist(double(i), \
									double(j), \
									double(row_id - 0.5), \
									double(col_id + 0.5));
					//	printf("%f\n", dist_0);		
						double dist_1 = calcDist(double(i), \
									double(j), \
									double(row_id - 0.5), \
									double(col_id));
						
					//	printf("%f\n", dist_1);		
						double dist_2 = calcDist(double(i), \
									double(j), \
									double(row_id - 0.5), \
									double(col_id - 0.5));
			
					//	printf("%f\n", dist_2);		
						if (dist_0 < temp_dist)
						{
					//		printf("dist_0\n");
							temp_dist = dist_0;
							face_id = 1;
							point_id = 0;

						}
						if (dist_1 < temp_dist)
						{
					//		printf("dist_1\n");
							temp_dist = dist_1;
							face_id = 1;
							point_id = 1;

						}
						
						if (dist_2 < temp_dist)
						{
					//		printf("dist_2\n");
							temp_dist = dist_2;
							face_id = 1;
							point_id = 2;

						}
					}
					
					// If has positive j boundary
					if (isKthBitSet(int(input[input_idx]), 5))
					{
					//	printf("Has positive j boundary. \n");
						double dist_0 = calcDist(double(i), \
									double(j), \
									double(row_id + 0.5), \
									double(col_id - 0.5));
						
						double dist_1 = calcDist(double(i), \
									double(j), \
									double(row_id + 0.5), \
									double(col_id));
						
						double dist_2 = calcDist(double(i), \
									double(j), \
									double(row_id + 0.5), \
									double(col_id + 0.5));
			
						if (dist_0 < temp_dist)
						{
							temp_dist = dist_0;
							face_id = 3;
							point_id = 0;

						}
						if (dist_1 < temp_dist)
						{
							temp_dist = dist_1;
							face_id = 3;
							point_id = 1;
						}
						
						if (dist_2 < temp_dist)
						{
							temp_dist = dist_2;
							face_id = 3;
							point_id = 2;

						}
					}
					//printf("\n");
						
					if (face_id == 0)
					{
						printf("(NEG_I, %d)  ", point_id);
					}
					if (face_id == 1)
					{
						printf("(NEG_J, %d)  ", point_id);
					}
					if (face_id == 2)
					{
						printf("(POS_I, %d)  ", point_id);
					}
					if (face_id == 3)
					{
						printf("(POS_J, %d)  ", point_id);
					}
				}
			}

			printf("\n");
		}
	}
}

double calcDist(double i, double j, double target_i, double target_j)
{
	double result = (i - target_i) * (i - target_i) + \
			(j - target_j) * (j - target_j);

	return sqrt(result);	
}

int check(int *ref, int *output, int length)
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
