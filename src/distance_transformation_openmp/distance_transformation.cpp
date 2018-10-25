#include "distance_transformation.h"

void distTransformation (char *scheme,
			int *FT, 
			uchar *raw_vol,
		       	float *sp2,	
			int height, int width, int depth, 
			float *ed_out)
{
	int slice_stride = height * width;

	int i, j, k;

	for (k = 0; k < depth; k++)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				int dep_id = FT[k * slice_stride + i * width + j] / slice_stride;
				
				int row_id = FT[k * slice_stride + i * width + j] \
						% slice_stride / width;

				int col_id = FT[k * slice_stride + i * width + j] \
					     	% slice_stride % width;
				
				if (row_id == i && col_id == j && k == dep_id)
				{
					ed_out[k * slice_stride + i * width + j] = 0.0;
				}
				else
				{
					if (strcmp(scheme, "--center-face") == 0)
					{
						ed_out[k * slice_stride + i * width + j] = \
						distToClosetFacePointOfCFV(
						sp2, \
						i, j, k, \
						row_id, col_id, dep_id, \
						raw_vol[FT[k * slice_stride + i * width + j]]); 
					}
					else if (strcmp(scheme, "--center-center") == 0)
					{
						ed_out[k * slice_stride + i * width + j] = \
						calcDist(
						sp2, \
						i, j, k, \
						row_id, col_id, dep_id); 
					}
				}
			}
		}
	}
}

double distToClosetFacePointOfCFV(
				float *sp2,
				int i, int j, int k,
				int cfv_i, int cfv_j, int cfv_k,
		       		uchar cfv_val)
{
	double dist = DBL_MAX;

	// If the CFV has negative i face
	if (isKthBitSet(cfv_val, 1))
	{
		double temp_dist = distToFacePoint(1,
					sp2,
					(double)i, (double)j, (double)k,
					(double)cfv_i, (double)cfv_j-0.5, (double)cfv_k);

		if (temp_dist < dist)
		{
			dist = temp_dist;
		}
	}

	// If the CFV has negative j face
	if (isKthBitSet(cfv_val, 2))
	{
		double temp_dist = distToFacePoint(0,
					sp2,
					(double)i, (double)j, (double)k,
					(double)cfv_i-0.5, (double)cfv_j, (double)cfv_k);

		if (temp_dist < dist)
		{
			dist = temp_dist;
		}
	}

	// If the CFV has negative k face
	if (isKthBitSet(cfv_val, 3))
	{
		double temp_dist = distToFacePoint(2,
					sp2,
					(double)i, (double)j, (double)k,
					(double)cfv_i, (double)cfv_j, (double)cfv_k-0.5);

		if (temp_dist < dist)
		{
			dist = temp_dist;
		}
	}

	// If the CFV has positive i face
	if (isKthBitSet(cfv_val, 4))
	{
		double temp_dist = distToFacePoint(1,
					sp2,
					(double)i, (double)j, (double)k,
					(double)cfv_i, (double)cfv_j+0.5, (double)cfv_k);

		if (temp_dist < dist)
		{
			dist = temp_dist;
		}
	}

	// If the CFV has positive j face
	if (isKthBitSet(cfv_val, 5))
	{
		double temp_dist = distToFacePoint(0,
					sp2,
					(double)i, (double)j, (double)k,
					(double)cfv_i+0.5, (double)cfv_j, (double)cfv_k);

		if (temp_dist < dist)
		{
			dist = temp_dist;
		}
	}

	// If the CFV has positive k face
	if (isKthBitSet(cfv_val, 6))
	{
		double temp_dist = distToFacePoint(2,
					sp2,
					(double)i, (double)j, (double)k,
					(double)cfv_i, (double)cfv_j, (double)cfv_k+0.5);

		if (temp_dist < dist)
		{
			dist = temp_dist;
		}
	}

	return dist;
}

double distToFacePoint(int unchanged, // which dimension stays unchanged
			float *sp2,
			double i, double j, double k, 
			double cfv_i, double cfv_j, double cfv_k)
{
	double dist = DBL_MAX;

	// When processing NEG/POS_J direction
	if (unchanged == 0)
	{
		double temp_j = cfv_j - 0.5;

		int add_j;
		for (add_j = 0; add_j < 3; add_j++)
		{
			double temp_k = cfv_k - 0.5;

			int add_k;
			for (add_k = 0; add_k < 3; add_k++)
			{
				double temp_dist = calcDist(
							sp2,
							i, j, k,
							cfv_i, 
							temp_j + 0.5 * add_j, 
							temp_k + 0.5 * add_k);
				
				if(temp_dist < dist)
				{
					dist = temp_dist;
				}

			}
		}
	}

	// When processing NEG/POS_I direction
	else if (unchanged == 1)
	{
		double temp_i = cfv_i - 0.5;

		int add_i;
		for (add_i = 0; add_i < 3; add_i++)
		{
			
			double temp_k = cfv_k - 0.5;

			int add_k;
			for (add_k = 0; add_k < 3; add_k++)
			{
				double temp_dist = calcDist(
							sp2,
							i, j, k,
							temp_i + 0.5 * add_i, 
							cfv_j, 
							temp_k + 0.5 * add_k);
				if(temp_dist < dist)
				{
					dist = temp_dist;
				}
			}
		}
	}

	// When processing NEG/POS_K direction	
	else if (unchanged == 2)
	{
		double temp_i = cfv_i - 0.5;

		int add_i;
		for (add_i = 0; add_i < 3; add_i++)
		{
			
			double temp_j = cfv_j - 0.5;

			int add_j;
			for (add_j = 0; add_j < 3; add_j++)
			{
				double temp_dist = calcDist(
							sp2,
							i, j, k,
							temp_i + 0.5 * add_i, 
							temp_j + 0.5 * add_j, 
							cfv_k);
				if(temp_dist < dist)
				{
					dist = temp_dist;
				}
			}
		}
	}
	
	return dist;
}

double calcDist(float *sp2,
		double i, double j, double k, \
		double target_i, double target_j, double target_k)
{
	double result = (i - target_i) * (i - target_i) * sp2[0] + \
			(j - target_j) * (j - target_j) * sp2[1] + \
			(k - target_k) * (k - target_k) * sp2[2];

	return sqrt(result);
}

bool isKthBitSet(int n, int k)
{
    if (n & (1 << (k - 1)))
        return 1;
    else
        return 0;
}
