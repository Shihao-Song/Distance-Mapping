#ifndef _DISTANCE_TRANSFORMATION_GPU_CUH_
#define _DISTANCE_TRANSFORMATION_GPU_CUH_

#include <math.h>
#include <float.h>

#include <stdio.h>

#include "omp.h"

#define NUM_THREADS_PER_BLOCK 512
#define NUM_BLOCKS_PER_GRID 512

typedef unsigned char uchar;

__global__ void distTransformation_GPU (int scheme,
                        uchar *raw_vol,
                        float sp2_0, float sp2_1, float sp2_2,
                        int height, int width, int depth,
                        double *ed_out);

__device__ double distToClosetFacePointOfCFV_GPU(
                                bool is_cfv,
                                float sp2_0, float sp2_1, float sp2_2,
                                int i, int j, int k,
                                int cfv_i, int cfv_j, int cfv_k,
                                uchar cfv_val);

__device__ double distToFacePoint_I_GPU(
                        float sp2_0, float sp2_1, float sp2_2,
                        double i, double j, double k,
                        double cfv_i, double cfv_j, double cfv_k);

__device__ double distToFacePoint_J_GPU(
                        float sp2_0, float sp2_1, float sp2_2,
                        double i, double j, double k,
                        double cfv_i, double cfv_j, double cfv_k);

__device__ double distToFacePoint_K_GPU(
                        float sp2_0, float sp2_1, float sp2_2,
                        double i, double j, double k,
                        double cfv_i, double cfv_j, double cfv_k);

__device__ double calcDist_GPU(float sp2_0, float sp2_1, float sp2_2,
                double i, double j, double k,
                double target_i, double target_j, double target_k);

__device__ bool isKthBitSet_GPU(int n, int k);

#endif
