#ifndef _MAURER_GPU_CUH_
#define _MAURER_GPU_CUH_

#include <float.h>

typedef unsigned char uchar;

#define D1_NUM_THREADS_PER_BLOCK_X 32
#define D1_NUM_THREADS_PER_BLOCK_Y 8
#define D1_NUM_THREADS_PER_BLOCK_Z 4

#define D2_NUM_THREADS_PER_BLOCK_X 8
#define D2_NUM_THREADS_PER_BLOCK_Y 32
#define D2_NUM_THREADS_PER_BLOCK_Z 4

#define D3_NUM_THREADS_PER_BLOCK_X 8
#define D3_NUM_THREADS_PER_BLOCK_Y 4
#define D3_NUM_THREADS_PER_BLOCK_Z 32

/*
	This function calculates the euclidean distance. 
*/
__device__ double ED(int vol_i, int vol_j, int vol_k, int fv, \
						int height, int width, int depth);


template <typename T>
__global__ void maurerFTGPU(T *input, int round, int height, int width, int depth, \
									int *dev_output);

#endif
