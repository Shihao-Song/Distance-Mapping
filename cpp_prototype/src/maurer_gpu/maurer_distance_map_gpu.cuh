#ifndef _MAURER_D_M_GPU_CUH_
#define _MAURER_D_M_GPU_CUH_

#include <cfloat>
#include <cstdlib>
#include <cmath>
#include <string>

#include "../tmp/Vol.h"

#include "../boundary_face_dist_gpu/boundary_face_distance_gpu.cuh"

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

using namespace std;

__device__ double ED_GPU(int vol_i, int vol_j, int vol_k,
                        int fv,
                        float sp2_0, float sp2_1, float sp2_2,
                        int height, int width, int depth);

template <typename T>
__global__ void maurerFT_GPU(T *input,
                        int round,
                        int height, int width, int depth,
                        float sp2_0, float sp2_1, float sp2_2,
                        double *dev_output);

class Maurer_Distance_Map_GPU
{
public:
        Maurer_Distance_Map_GPU (char *mapping_scheme)
        {
		boundary_face_dist_calc_gpu = new Boundary_Face_Distance_GPU(mapping_scheme);
        }

        ~Maurer_Distance_Map_GPU ()
        {
		free(boundary_face_dist_calc_gpu);
        }

        void run_maurer_gpu(Vol *vol)
	{
		// Volume information
		int HEIGHT = vol->height;
		int WIDTH = vol->width;
		int DEPTH = vol->depth;

		float *sp2 = vol->sp2;

		unsigned char *raw_vol = vol->raw_vol;

		// Initialize GPU output
		if (cudaHostAlloc((void **)&dist_mapping_maurer_gpu,
				HEIGHT * WIDTH * DEPTH * sizeof(double),
				cudaHostAllocDefault) != cudaSuccess)
		{
			printf("cudaHostAlloc() failed! \n");
			exit(0);
		}

		/*******************************************************
			Allocate device (GPU) memory
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

		/*******************************************************
			Distance mapping using GPU
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
	
		boundary_face_dist_calc_gpu->run_boundary_face_distance_calculation_gpu(
					HEIGHT, WIDTH, DEPTH,
					sp2[0], sp2[1], sp2[2],
					dev_raw_vol,
					dev_dist_mapping_maurer_gpu[0]);	


		if (cudaMemcpy(dist_mapping_maurer_gpu, dev_dist_mapping_maurer_gpu[0],
				HEIGHT * WIDTH * DEPTH * sizeof(double),
				cudaMemcpyDeviceToHost) != cudaSuccess)
		{
			printf("cudaMemcpy() failed! \n");
			exit(0);
		}

		cudaFree(dev_raw_vol);
		
		for (int i = 0; i < 2; i++)
		{
			cudaFree(dev_dist_mapping_maurer_gpu[i]);
		}	
	}

        double *dist_mapping_maurer_gpu;

protected:

	Boundary_Face_Distance_GPU *boundary_face_dist_calc_gpu;
};

__device__ double ED_GPU(int vol_i, int vol_j, int vol_k, 
			int fv,
			float sp2_0, float sp2_1, float sp2_2,
			int height, int width, int depth)
{
	int vol_slice_stride = height * width;

	int fv_k = fv / vol_slice_stride;
	int fv_i = (fv % vol_slice_stride) / width;
	int fv_j = (fv % vol_slice_stride) % width;

	double temp = 0;
	temp = (fv_i - vol_i) * (fv_i - vol_i) * sp2_0 +
			(fv_j - vol_j) * (fv_j - vol_j) * sp2_1 +
			(fv_k - vol_k) * (fv_k - vol_k) * sp2_2;

	return sqrt((double)temp);
}

template <typename T>
__global__ void maurerFT_GPU(T *input, 
			int round, 
			int height, int width, int depth,
			float sp2_0, float sp2_1, float sp2_2,
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
					partial_voxels[threadIds[2] * shared_dist_betw_slices + threadIds[1] * blockDims[0] + threadIds[0]] =
					(input[loading_pos[2] * vol_dist_betw_slices + loading_pos[1] * vol_dim[0] + loading_pos[0]] != 0) ?
					loading_pos[2] * vol_dist_betw_slices + loading_pos[1] * vol_dim[0] + loading_pos[0] : -1;
				}
			}
			else
			{	
				if (loading_pos[0] < vol_dim[0] && loading_pos[1] < vol_dim[1] && loading_pos[2] < vol_dim[2])
				{
					partial_voxels[threadIds[2] * shared_dist_betw_slices + threadIds[1] * blockDims[0] + threadIds[0]] =
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
					fv_index = partial_voxels[threadIds[2] * shared_dist_betw_slices +
												threadIds[1] * blockDims[0] + i];
				}
				else if (round == 1)
				{
					fv_index = partial_voxels[threadIds[2] * shared_dist_betw_slices +
												i * blockDims[0] + threadIds[0]];
				}
				else
				{
					fv_index = partial_voxels[i * shared_dist_betw_slices +
												threadIds[1] * blockDims[0] + threadIds[0]];
				}

				if(fv_index != -1)
				{
					double tempDist = ED_GPU(thread_pos[1], thread_pos[0], thread_pos[2], fv_index,
													sp2_0, sp2_1, sp2_2,
													height, width, depth);
					
					if (tempDist < minDist)
					{
						minDist = tempDist;
						cfv_index = fv_index;
					}
				}
			}

			__syncthreads(); // This may not necessary
			
			if (partial_cfv_valid == 0)
			{
				partial_cfv[threadIds[2] * shared_dist_betw_slices 
								+ threadIds[1] * blockDims[0] + threadIds[0]] = cfv_index;

				partial_cfv_valid = 1;
			}
			else
			{
				int current_cfv_index = partial_cfv[threadIds[2] * shared_dist_betw_slices
										+ threadIds[1] * blockDims[0] + threadIds[0]];

				if (cfv_index != -1)
				{
					if (current_cfv_index == -1)
					{
						partial_cfv[threadIds[2] * shared_dist_betw_slices
								+ threadIds[1] * blockDims[0] + threadIds[0]] = cfv_index;
					}
					else
					{
						double currentDist = ED_GPU(thread_pos[1], thread_pos[0], thread_pos[2], current_cfv_index,
															sp2_0, sp2_1, sp2_2,
															height, width, depth);

						if (minDist < currentDist)
						{
							partial_cfv[threadIds[2] * shared_dist_betw_slices
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
						+ thread_pos[0]] = double(partial_cfv[threadIds[2] * shared_dist_betw_slices
									+ threadIds[1] * blockDims[0] + threadIds[0]]);
		}
		/* 
			direction and stride for thread block movement 
				(1) when round = 0, thread blocks are moving along the rows;
				(2) when round = 1, thread blocks are moving along the columns;
				(3) when round = 2, thread blocks are moving along the depth.
		*/
		thread_pos[round] += blockDims[round];
		
		__syncthreads();
	}
//	}
}

#endif
