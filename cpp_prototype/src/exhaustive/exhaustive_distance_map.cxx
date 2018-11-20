#include "exhaustive_distance_map.h"

void Exhaustive_Distance_Map::run_exhaustive(Vol *vol)
{
	// Volume information
	int height = vol->height;
	int width = vol->width;
	int depth = vol->depth;

	float *sp2 = vol->sp2;

	unsigned char *raw_vol = vol->raw_vol;

	// Intialize distance mapping outputs
	dist_mapping_exhaustive =
                                (double *)malloc(height * width * depth * sizeof(double));

	for (int i = 0; i < height * width * depth; i++)
        {
                dist_mapping_exhaustive[i] = -1.0;
        }

	// Perform FT using Exhaustive
	exhaustiveFT(raw_vol, sp2,
                	height, width, depth,
                	dist_mapping_exhaustive);

	boundary_face_dist_calc->run_boundary_face_distance_calculation(vol, 
									dist_mapping_exhaustive);
//	free(dist_mapping_exhaustive);
}

void Exhaustive_Distance_Map::exhaustiveFT(unsigned char *vol, float *sp2,
                				int height, int width, int depth,
                				double *output)
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
                                exhaustiveSearch(vol, sp2,
                                                i, j, k,
                                                height, width, depth,
                                                output);
                        }
                }
        }
}

void Exhaustive_Distance_Map::exhaustiveSearch(unsigned char *vol, float *sp2,
                				int vol_i, int vol_j, int vol_k,
                				int height, int width, int depth,
                				double *output)
{
        double minDist = DBL_MAX;

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
                                        double tempDist = (i - vol_i) * (i - vol_i) * sp2[0] + \
                                                        (j - vol_j) * (j - vol_j) * sp2[1] + \
                                                        (k - vol_k) * (k - vol_k) * sp2[2];

                                        if (tempDist < minDist)
                                        {
                                                minDist = tempDist;

                                                cfv_i = i;
                                                cfv_j = j;
                                                cfv_k = k;
                                        }
                                }
                        }
                }
        }
        
	output[vol_k * slice_stride + vol_i * width + vol_j] =
                double(cfv_k * slice_stride + cfv_i * width + cfv_j);
}
