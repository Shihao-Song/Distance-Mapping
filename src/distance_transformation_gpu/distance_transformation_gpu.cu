#include "distance_transformation_gpu.cuh"

__global__ void distTransformation_GPU (int scheme,
			uchar *raw_vol,
		       	float sp2_0, float sp2_1, float sp2_2,	
			int height, int width, int depth, 
			double *ed_out)
{
	int size_of_vol = height * width * depth;

	int slice_stride = height * width;

	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	while (tid < size_of_vol)
	{
		int cfv = int(ed_out[tid]); 

		// Coordinates of CFV
		int cfv_k = cfv / slice_stride;
		int cfv_i = cfv % slice_stride / width;
               	int cfv_j = cfv % slice_stride % width;	
	
		// Coordinates of me
		int k = tid / slice_stride;
		int i = tid % slice_stride / width;
		int j = tid % slice_stride % width;

		// Am myself the CFV?
		bool is_cfv = (tid == cfv) ? 1 : 0;

		// Distance translation
		if (scheme == 0)
		{
			// center-to-center
			ed_out[tid] = calcDist_GPU( sp2_0, sp2_1, sp2_2, \
                                                i, j, k, \
                                                cfv_i, cfv_j, cfv_k);
		}
		else if (scheme == 1)
		{
			// center-to-face
			ed_out[tid] = distToClosetFacePointOfCFV_GPU( \
					 	is_cfv, \
                                                sp2_0, sp2_1, sp2_2, \
                                                i, j, k, \
                                                cfv_i, cfv_j, cfv_k, \
                                                raw_vol[cfv]);
		}	

		tid += blockDim.x * gridDim.x;
	}	
}

__device__ double distToClosetFacePointOfCFV_GPU(
				bool is_cfv,
				float sp2_0, float sp2_1, float sp2_2,
				int i, int j, int k,
				int cfv_i, int cfv_j, int cfv_k,
		       		uchar cfv_val)
{
	
	double closet_dist = (is_cfv == 1) ? 0.0 : DBL_MAX;

	double calculated_dists[6];

	// Negative i face
	calculated_dists[0] = (isKthBitSet_GPU(cfv_val, 1) == 0) ? DBL_MAX : \
					distToFacePoint_I_GPU( \
                                        sp2_0, sp2_1, sp2_2, \
                                        (double)i, (double)j, (double)k, \
                                        (double)cfv_i, (double)cfv_j-0.5, (double)cfv_k);
	
	// Negative j face
	calculated_dists[1] = (isKthBitSet_GPU(cfv_val, 2) == 0) ? DBL_MAX : \
					distToFacePoint_J_GPU( \
                                        sp2_0, sp2_1, sp2_2, \
                                        (double)i, (double)j, (double)k, \
                                        (double)cfv_i-0.5, (double)cfv_j, (double)cfv_k);
	
	// Negative k face
	calculated_dists[2] = (isKthBitSet_GPU(cfv_val, 3) == 0) ? DBL_MAX : \
					distToFacePoint_K_GPU( \
                                        sp2_0, sp2_1, sp2_2, \
                                        (double)i, (double)j, (double)k, \
                                        (double)cfv_i, (double)cfv_j, (double)cfv_k-0.5);
	
	// Positive i face
	calculated_dists[3] = (isKthBitSet_GPU(cfv_val, 4) == 0) ? DBL_MAX : \
					distToFacePoint_I_GPU( \
                                        sp2_0, sp2_1, sp2_2, \
                                        (double)i, (double)j, (double)k, \
                                        (double)cfv_i, (double)cfv_j+0.5, (double)cfv_k);

	// Positive j face
	calculated_dists[4] = (isKthBitSet_GPU(cfv_val, 5) == 0) ? DBL_MAX : \
					distToFacePoint_J_GPU( \
                                        sp2_0, sp2_1, sp2_2, \
                                        (double)i, (double)j, (double)k, \
                                        (double)cfv_i+0.5, (double)cfv_j, (double)cfv_k);

	// Positive k face
	calculated_dists[5] = (isKthBitSet_GPU(cfv_val, 6) == 0) ? DBL_MAX : \
					distToFacePoint_K_GPU( \
                                        sp2_0, sp2_1, sp2_2, \
                                        (double)i, (double)j, (double)k, \
                                        (double)cfv_i, (double)cfv_j, (double)cfv_k+0.5);
	
	for (int i = 0; i < 6; i++)
	{
		if (calculated_dists[i] < closet_dist)
		{
			closet_dist = calculated_dists[i];
		}
	}

	
	return closet_dist;
}

__device__ double distToFacePoint_I_GPU(
			float sp2_0, float sp2_1, float sp2_2,
                        double i, double j, double k,
                        double cfv_i, double cfv_j, double cfv_k)
{	
	double dist = DBL_MAX;

	double temp_i = cfv_i - 0.5;

	int add_i;
	for (add_i = 0; add_i < 3; add_i++)
	{
			
		double temp_k = cfv_k - 0.5;

		int add_k;
		for (add_k = 0; add_k < 3; add_k++)
		{
			double temp_dist = calcDist_GPU(
						sp2_0, sp2_1, sp2_2, \
						i, j, k, \
						temp_i + 0.5 * add_i, \
						cfv_j, \
						temp_k + 0.5 * add_k);
			if(temp_dist < dist)
			{
				dist = temp_dist;
			}
		}
	}

	return dist;
}

__device__ double distToFacePoint_J_GPU(
			float sp2_0, float sp2_1, float sp2_2,
                        double i, double j, double k,
                        double cfv_i, double cfv_j, double cfv_k)
{
	double dist = DBL_MAX;

	double temp_j = cfv_j - 0.5;

	int add_j;
	for (add_j = 0; add_j < 3; add_j++)
	{
		double temp_k = cfv_k - 0.5;

		int add_k;
		for (add_k = 0; add_k < 3; add_k++)
		{
			double temp_dist = calcDist_GPU(
						sp2_0, sp2_1, sp2_2, \
						i, j, k, \
						cfv_i, \
						temp_j + 0.5 * add_j, \
						temp_k + 0.5 * add_k);
				
			if(temp_dist < dist)
			{
				dist = temp_dist;
			}

		}
	}
	
	return dist;
}

__device__ double distToFacePoint_K_GPU(
			float sp2_0, float sp2_1, float sp2_2,
                        double i, double j, double k,
                        double cfv_i, double cfv_j, double cfv_k)
{
	double dist = DBL_MAX;

	double temp_i = cfv_i - 0.5;

	int add_i;
	for (add_i = 0; add_i < 3; add_i++)
	{
			
		double temp_j = cfv_j - 0.5;

		int add_j;
		for (add_j = 0; add_j < 3; add_j++)
		{
			double temp_dist = calcDist_GPU(
						sp2_0, sp2_1, sp2_2, \
						i, j, k, \
						temp_i + 0.5 * add_i, \
						temp_j + 0.5 * add_j, \
						cfv_k);
				
			if(temp_dist < dist)
			{
				dist = temp_dist;
			}
		}
	}

	return dist;
}

__device__ double calcDist_GPU(float sp2_0, float sp2_1, float sp2_2,
		double i, double j, double k,
		double target_i, double target_j, double target_k)
{
	double result = (i - target_i) * (i - target_i) * sp2_0 + \
			(j - target_j) * (j - target_j) * sp2_1 + \
			(k - target_k) * (k - target_k) * sp2_2;

	return sqrt(result);
}

__device__ bool isKthBitSet_GPU(int n, int k)
{
    if (n & (1 << (k - 1)))
        return 1;
    else
        return 0;
}

