#include "maurer_distance_map.h"

void Maurer_Distance_Map::run_maurer(Vol *vol)
{
	// Volume information
	int height = vol->height;
	int width = vol->width;
	int depth = vol->depth;

	float *sp2 = vol->sp2;

	unsigned char *raw_vol = vol->raw_vol;

	// Intialize distance mapping outputs
	double *dist_mapping_maurer_openmp =
                                (double *)malloc(height * width * depth * sizeof(double));

	for (int i = 0; i < height * width * depth; i++)
        {
                dist_mapping_maurer_openmp[i] = -1.0;
        }

	// Perform FT using Maurer's
	maurerFT(raw_vol, 
		sp2,
                height, width, depth,
                dist_mapping_maurer_openmp);

	free(dist_mapping_maurer_openmp);

}

void Maurer_Distance_Map::maurerFT(unsigned char *vol,
        				float *sp2,
        				int height, int width, int depth,
        				double *output)
{
        int dim;

        for (dim = 1; dim <= 3; dim++)
        {
                voronoiFT(dim, vol, sp2, height, width, depth, output);
        }
}

void Maurer_Distance_Map::voronoiFT(int dim,
        				unsigned char *vol,
        				float *sp2,
        				int height, int width, int depth,
        				double *output)
{
        switch (dim)
        {
                case 1:
                        runVoronoiFT1D(vol, sp2, height, width, depth, output);
                        break;
                case 2:
                        runVoronoiFT2D(sp2, height, width, depth, output);
                        break;
                case 3:
                        runVoronoiFT3D(sp2, height, width, depth, output);
                        break;
                default:
                        break;
        }
}

void Maurer_Distance_Map::runVoronoiFT1D(unsigned char *vol,
                                		float *sp2,
                                		int height, int width, int depth,
                                		double *output)
{
	// GNodes
	GNodes g;

	// Distance between slices
	int slice_stride = height * width;

	int k;

	#pragma omp parallel shared(vol, sp2, output) private(g, k)
	{
	g.size = 0;
	g.max_size = width;
	g.nodes = (node *)malloc(g.max_size * sizeof(node));

	#pragma omp for
	for (k = 0; k < depth; k++)
	{
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int fv = int(vol[k * slice_stride + i * width + j]); 

				if (fv != 0)
				{
					push(&g, i, j, k);
				}
			}

			if (g.size == 0)
			{
				continue;
			}

			// Query partial voronoi diagram
			for (int j = 0; j < width; j++)
			{
				int ite = 0;

				while ((ite < (g.size - 1)) &&
					(ED(sp2, i, j, k, &(g.nodes[ite])) >
					ED(sp2, i, j, k, &(g.nodes[ite+1]))))
				{
					ite++;
				}

				output[k * slice_stride + i * width + j] =
					g.nodes[ite].fv_pos[2] * slice_stride +
					g.nodes[ite].fv_pos[0] * width +
					g.nodes[ite].fv_pos[1];
			}
			g.size = 0;
		}
	}
	free(g.nodes);
	}
}

void Maurer_Distance_Map::runVoronoiFT2D(float *sp2, 
						int height, int width, int depth,
                                		double *vol)
{
	GNodes g;
	
	int Rd[3];
	int w[3];

	// Distance between slices
	int slice_stride = height * width;

	int k;
	#pragma omp parallel shared(sp2, vol) private(g, Rd, w, k)
    	{
	g.size = 0;
	g.max_size = height;
	g.nodes = (node *)malloc(g.max_size * sizeof(node));

	#pragma omp for
	for (k = 0; k < depth; k++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int i = 0; i < height; i++)
			{
				int fv = int(vol[k * slice_stride + i * width + j]);

				if (fv != -1.0)
				{
					int fv_k = fv / slice_stride;
					
					int fv_i = (fv % slice_stride) / width;
					
					int fv_j = (fv % slice_stride) % width;
					
					if(g.size < 2)
					{
						push(&g, fv_i, fv_j, fv_k);
					}
					else
					{
						w[0] = fv_i;
						w[1] = fv_j;
						w[2] = fv_k;

						Rd[0] = i;
						Rd[1] = j;
						Rd[2] = k;

						while (g.size >= 2 && 
							removeFT2D(sp2, &g, w, Rd))
						{
							pop(&g);
						}

						push(&g, fv_i, fv_j, fv_k);
					}
				}
			}

			if (g.size == 0)
			{
				continue;
			}
			
			// Query partial voronoi diagram
			for (int i = 0; i < height; i++)
			{
				double minDist = DBL_MAX;
				int minIndex = -1;

				int ite = 0;
				while(ite < g.size)
				{
					double tempDist = 
						ED(sp2, i, j, k, &(g.nodes[ite]));

					if(tempDist < minDist)
					{
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] =
					g.nodes[minIndex].fv_pos[2] * slice_stride +
					g.nodes[minIndex].fv_pos[0] * width +
					g.nodes[minIndex].fv_pos[1];
			}
			g.size = 0;
		}
	}
	free(g.nodes);
	}
}

void Maurer_Distance_Map::runVoronoiFT3D(float *sp2,
                                		int height, int width, int depth,
                                		double *vol)
{
	GNodes g;

	int Rd[3];
	int w[3];

	// Distance between slices
	int slice_stride = height * width;

	int i;
	#pragma omp parallel shared(sp2, vol) private(g, Rd, w, i)
    	{
	g.size = 0;
	g.max_size = depth;
	g.nodes = (node *)malloc(g.max_size * sizeof(node));

	#pragma omp for
	for (i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			for (int k = 0; k < depth; k++)
			{
				int fv = int(vol[k * slice_stride + i * width + j]); 
				
				if (fv != -1.0)
				{
					int fv_k = fv / slice_stride;

					int fv_i = (fv % slice_stride) / width;

					int fv_j = (fv % slice_stride) % width;
					
					if(g.size < 2)
					{
						push(&g, fv_i, fv_j, fv_k);
					}
					else
					{
						w[0] = fv_i;
						w[1] = fv_j;
						w[2] = fv_k;

						Rd[0] = i;
						Rd[1] = j;
						Rd[2] = k;

						while (g.size >= 2 &&
							removeFT3D(sp2, &g, w, Rd))
						{
							pop(&g);
						}

						push(&g, fv_i, fv_j, fv_k);
					}
				}
			}
				
			if (g.size == 0)
			{
				continue;
			}

			// Query partial voronoi diagram
			for (int k = 0; k < depth; k++)
			{
				double minDist = DBL_MAX;
				int minIndex = -1;

				int ite = 0;
				while(ite < g.size)
				{
					double tempDist = 
						ED(sp2, i, j, k, &(g.nodes[ite]));

					if(tempDist < minDist)
					{
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] =
					g.nodes[minIndex].fv_pos[2] * slice_stride +
					g.nodes[minIndex].fv_pos[0] * width +
					g.nodes[minIndex].fv_pos[1];
			}
			g.size = 0;
		}
	}
	free(g.nodes);
	}
}

int Maurer_Distance_Map::removeFT2D(float *sp2,
                        		GNodes *g_nodes, int *w, int *Rd)
{
	node u = g_nodes->nodes[g_nodes->size - 2];
	node v = g_nodes->nodes[g_nodes->size - 1];

	double a = (v.fv_pos[0] - u.fv_pos[0]) * sqrt(sp2[0]);
	double b = (w[0] - v.fv_pos[0]) * sqrt(sp2[0]);
	double c = a + b;

	double vRd = 0.0;
	double uRd = 0.0;
	double wRd = 0.0;

	for (int i = 1; i < 3; i++)
	{
		vRd += (v.fv_pos[i] - Rd[i]) * (v.fv_pos[i] - Rd[i]) * sp2[i];
		uRd += (u.fv_pos[i] - Rd[i]) * (u.fv_pos[i] - Rd[i]) * sp2[i];
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]) * sp2[i];
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0.0);
}

int Maurer_Distance_Map::removeFT3D(float *sp2,
                        		GNodes *g_nodes, int *w, int *Rd)
{
	node u = g_nodes->nodes[g_nodes->size - 2];
	node v = g_nodes->nodes[g_nodes->size - 1];

	double a = (v.fv_pos[2] - u.fv_pos[2]) * sqrt(sp2[2]);
	double b = (w[2] - v.fv_pos[2]) * sqrt(sp2[2]);
	double c = a + b;

	double vRd = 0;
	double uRd = 0;
	double wRd = 0;

	for (int i = 0; i < 2; i++)
	{
		vRd += (v.fv_pos[i] - Rd[i]) * (v.fv_pos[i] - Rd[i]) * sp2[i];
		uRd += (u.fv_pos[i] - Rd[i]) * (u.fv_pos[i] - Rd[i]) * sp2[i];
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]) * sp2[i];
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0.0);
}

double Maurer_Distance_Map::ED(float *sp2,
				int vol_i, int vol_j, int vol_k,
                        	node *fv)
{
	double dist = 0;

        dist = (fv->fv_pos[0] - vol_i) * (fv->fv_pos[0] - vol_i) * sp2[0] +
		(fv->fv_pos[1] - vol_j) * (fv->fv_pos[1] - vol_j) * sp2[1] +
		(fv->fv_pos[2] - vol_k) * (fv->fv_pos[2] - vol_k) * sp2[2];

	return sqrt(dist);
}

