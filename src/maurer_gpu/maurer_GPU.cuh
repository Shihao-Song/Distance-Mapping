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

#define SIZE_OF_SHARED_MEMORY 1024

__device__ double ED(int vol_i, int vol_j, int vol_k, int fv, \
						int height, int width, int depth)
{
	int vol_slice_stride = height * width;

	int fv_k = fv / vol_slice_stride;
	int fv_i = (fv % vol_slice_stride) / width;
	int fv_j = (fv % vol_slice_stride) % width;

	int temp = 0;
	temp = (fv_i - vol_i) * (fv_i - vol_i) + \
			(fv_j - vol_j) * (fv_j - vol_j) + \
			(fv_k - vol_k) * (fv_k - vol_k);

	return sqrt((double)temp);
}

template <typename T>
__global__ void maurerFTGPU(T *input, int round, int height, int width, int depth, \
						float sp2_0, float sp2_1, float sp2_2, \
						double *dev_output)
{
	/* partial_cfv contains the calculated partial feature voxels */
	__shared__ int partial_cfv[SIZE_OF_SHARED_MEMORY];

	/* partial_voxels provides voxels to calculate partial_cfv */
	__shared__ int partial_voxels[SIZE_OF_SHARED_MEMORY]; 

	/*
		Put all threadIds, blockIds, blockDim.*, gridDim.* and volume dims 
		into an arrays to simplify future processing
	*/
	int threadIds[3];
	threadIds[0] = threadIdx.x;
	threadIds[1] = threadIdx.y;
	threadIds[2] = threadIdx.z;

	int blockIds[3];
	blockIds[0] = blockIdx.x;
	blockIds[1] = blockIdx.y;
	blockIds[2] = blockIdx.z;

	int blockDims[3];
	blockDims[0] = blockDim.x;
	blockDims[1] = blockDim.y;
	blockDims[2] = blockDim.z;
	
	// vol_dim: 0 - number of columns, 1 - number of rows, 2 - number of slices
	int vol_dim[3];
	vol_dim[0] = width;
	vol_dim[1] = height;
	vol_dim[2] = depth;

	/*	
		Initial positions of thread location, the indexing stride 
		should be applied to these values

		thread_pos: 0 - col_id, 1 - row_id, 2 - dep_id;
	*/
	int thread_pos[3];
	thread_pos[2] = threadIds[2] + blockIds[2] * blockDims[2];
	thread_pos[1] = threadIds[1] + blockIds[1] * blockDims[1];
	thread_pos[0] = threadIds[0] + blockIds[0] * blockDims[0];

	/*
		Since our volume is constructed with row major policy, a common way
		to address a voxel in a row-major colume (x, y, z) is:
		z * (# rows * # columns) + y * #columns + x, where # rows * # columns 
		can be abbreviated as vol_dist_betw_slices (distance between volume slices)

		shared_dist_betw_slices is used to address partial volumes stored in
		the shared memory region
	*/
	int vol_dist_betw_slices = vol_dim[0] * vol_dim[1];
	int shared_dist_betw_slices = blockDims[0] * blockDims[1];

	/* cfv calculation starts here */
	int loading_pos[3];
	loading_pos[0] = thread_pos[0];
	loading_pos[1] = thread_pos[1];
	loading_pos[2] = thread_pos[2];
	
	int limit = int(ceil((double)vol_dim[round] / (double)blockDims[round]));
	for (int i = 0; i < limit; i++)
	{
		/*
			Initialize shared memory region 
		*/
		partial_cfv[threadIds[2] * shared_dist_betw_slices + threadIds[1] * blockDims[0] + threadIds[0]] = -1;
        	partial_voxels[threadIds[2] * shared_dist_betw_slices + threadIds[1] * blockDims[0] + threadIds[0]] = -1;

		__syncthreads();

		// Loading from the very beginning
		loading_pos[round] = threadIds[round];

		// This indicate that contents in partial_cfv is currently valid or not
		int partial_cfv_valid = 0; 
		
		for (int j = 0; j < limit; j++)
		//while (loading_pos[round] < vol_dim[round])
		{
			/*
				Step One: load partial voxels
			*/
			if (round == 0)
			{
				if (loading_pos[0] < vol_dim[0] && loading_pos[1] < vol_dim[1] && loading_pos[2] < vol_dim[2])
				{
					partial_voxels[threadIds[2] * shared_dist_betw_slices + threadIds[1] * blockDims[0] + threadIds[0]] = \
					(input[loading_pos[2] * vol_dist_betw_slices + loading_pos[1] * vol_dim[0] + loading_pos[0]] != 0) ? \
					loading_pos[2] * vol_dist_betw_slices + loading_pos[1] * vol_dim[0] + loading_pos[0] : -1;
				}
			}
			else
			{	
				if (loading_pos[0] < vol_dim[0] && loading_pos[1] < vol_dim[1] && loading_pos[2] < vol_dim[2])
				{
					partial_voxels[threadIds[2] * shared_dist_betw_slices + threadIds[1] * blockDims[0] + threadIds[0]] = \
					int(input[loading_pos[2] * vol_dist_betw_slices + loading_pos[1] * vol_dim[0] + loading_pos[0]]);
				}
			}

			__syncthreads();

			/*
				Step Two: compute partial cfv
			*/
			int fv_index = -1;
			double minDist = DBL_MAX;
			int cfv_index = -1;

			for (int i = 0; i < blockDims[round]; i++)
			{	
				if (round == 0)
				{
					fv_index = partial_voxels[threadIds[2] * shared_dist_betw_slices + \
												threadIds[1] * blockDims[0] + i];
				}
				else if (round == 1)
				{
					fv_index = partial_voxels[threadIds[2] * shared_dist_betw_slices + \
												i * blockDims[0] + threadIds[0]];
				}
				else
				{
					fv_index = partial_voxels[i * shared_dist_betw_slices + \
												threadIds[1] * blockDims[0] + threadIds[0]];
				}

				if(fv_index != -1)
				{
					double tempDist = ED(thread_pos[1], thread_pos[0], thread_pos[2], \
										fv_index, height, width, depth);
					
					if (tempDist < minDist)
					{
						minDist = tempDist;
						cfv_index = fv_index;
					}
				}
			}

			__syncthreads();
			
			if (partial_cfv_valid == 0)
			{
				partial_cfv[threadIds[2] * shared_dist_betw_slices \
								+ threadIds[1] * blockDims[0] + threadIds[0]] = cfv_index;

				partial_cfv_valid = 1;
			}
			else
			{
				int current_cfv_index = partial_cfv[threadIds[2] * shared_dist_betw_slices \
										+ threadIds[1] * blockDims[0] + threadIds[0]];

				if (cfv_index != -1)
				{
					if (current_cfv_index == -1)
					{
						partial_cfv[threadIds[2] * shared_dist_betw_slices \
								+ threadIds[1] * blockDims[0] + threadIds[0]] = cfv_index;
					}
					else
					{
						double currentDist = ED(thread_pos[1], thread_pos[0], thread_pos[2], \
											current_cfv_index, height, width, depth);

						if (minDist < currentDist)
						{
							partial_cfv[threadIds[2] * shared_dist_betw_slices \
								+ threadIds[1] * blockDims[0] + threadIds[0]] = cfv_index;
						}
					}
				}
			}

			/* 
				direction and stride for thread block movement 
					(1) when round = 0, thread blocks are moving along the rows;
					(2) when round = 1, thread blocks are moving along the columns;
					(3) when round = 2, thread blocks are moving along the depth.
			*/

			loading_pos[round] += blockDims[round];
			
			__syncthreads();
		}
	
		if (thread_pos[0] < vol_dim[0] && thread_pos[1] < vol_dim[1] && thread_pos[2] < vol_dim[2])
		{
			dev_output[thread_pos[2] * vol_dist_betw_slices + thread_pos[1] * vol_dim[0] 
						+ thread_pos[0]] = double(partial_cfv[threadIds[2] * shared_dist_betw_slices \
									+ threadIds[1] * blockDims[0] + threadIds[0]]);
		}
		/* 
			direction and stride for thread block movement 
				(1) when round = 0, thread blocks are moving along the rows;
				(2) when round = 1, thread blocks are moving along the columns;
				(3) when round = 2, thread blocks are moving along the depth.
		*/
		thread_pos[round] += blockDims[round];
	}
//	}
}

#endif
