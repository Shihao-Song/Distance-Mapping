#include "volume.h"
#include "exhaustiveFT.h"

void exhaustiveFT(uchar *vol, int height, int width, int depth, \
	int *output)
{
	// Distance between slices
	int slice_stride = height * width;

	int i, j, k;
	for (k = 0; k < depth; k++)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				exhaustiveSearch(i, j, k, vol, \
					height, width, depth, output);
			}
		}
	}
}

void exhaustiveSearch(int vol_i, int vol_j, int vol_k, uchar *vol, \
	int height, int width, int depth, int *output)
{

	//int init = 1;
	int minDist = INT_MAX;

	// Following variables are used to record the positions of
	// the closet feature voxel
	int cfv_i, cfv_j, cfv_k;

	// Distance between slices
	int slice_stride = height * width;

	int i, j, k;
	for (k = 0; k < depth; k++)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				if (vol[k * slice_stride + i * width + j] != 0)
				{
					/*
					if (init == 1)
					{
						// Initialize the minDist to the distance between
						// input voxel to the the first feature voxel it
						// encounters
						minDist = (i - vol_i) * (i - vol_i) + \
								(j - vol_j) * (j - vol_j) + \
								(k - vol_k) * (k - vol_k);

						init = 0;

						cfv_i = i;
						cfv_j = j;
						cfv_k = k;
					}
					else
					{
						*/
						int tempDist = (i - vol_i) * (i - vol_i) + \
								(j - vol_j) * (j - vol_j) + \
								(k - vol_k) * (k - vol_k);

						if (tempDist < minDist)
						{
							minDist = tempDist;

							cfv_i = i;
							cfv_j = j;
							cfv_k = k;
						}
					//}
				}
			}
		}
	}
	output[vol_k * slice_stride + vol_i * width + vol_j] = \
		cfv_k * slice_stride + cfv_i * width + cfv_j;
}
