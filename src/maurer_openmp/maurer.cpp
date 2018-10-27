#include "maurer.h"

void maurerFT(uchar *vol, float *sp2, \
	int height, int width, int depth, \
	double *output)
{
	int dim;

	for (dim = 1; dim <= 3; dim++)
	{
		VoronoiFT(dim, vol, sp2, height, width, depth, output);
	}
}

void VoronoiFT(int dim, \
	uchar *vol, float *sp2, \
	int height, int width, int depth, \
	double *output)
{
	switch (dim)
	{
		case 1:
			RunVoronoiFT1D(vol, sp2, height, width, depth, output);
			break;
		case 2:
//			RunVoronoiFT2D(sp2, height, width, depth, output);
			break;
		case 3:
//			RunVoronoiFT3D(sp2, height, width, depth, output);
			break;
		default:
			break;
	}
}

void RunVoronoiFT1D(uchar *vol, float *sp2, \
		int height, int width, int depth,\
		double *output)
{

	GNodes g;
	
	// Distance between slices
	int slice_stride = height * width;

	int k;

	#pragma omp parallel shared(vol, sp2, output) private(g,k)
	{
	g.size = 0;
        g.max_size = width;
        g.stack = (node *)malloc(g.max_size * sizeof(node));

	#pragma omp for
	for (k = 0; k < depth; k++)
	{
		int i;
		for (i = 0; i < height; i++)
		{
			int j;
			for (j = 0; j < width; j++)
			{
				if (vol[k * slice_stride + i * width + j] != 0)
				{
					push(&g, i, j, k);
				}
			}

			if (g.size == 0)
			{
				continue;
			}

			// Query partial voronoi diagram
			for (j = 0; j < width; j++)
			{
				int ite = 0;
				while ((ite < (g.size - 1)) && \
					(ED(sp2, i, j, k, &(g.stack[ite])) > \
					ED(sp2, i, j, k, &(g.stack[ite+1]))))
				{
					ite++;
				}

				output[k * slice_stride + i * width + j] = \
					double(g.stack[ite].fv_pos[2] * slice_stride + \
					g.stack[ite].fv_pos[0] * width + \
					g.stack[ite].fv_pos[1]);
			}
			g.size = 0;
		}
	}
	free(g.stack);
	}
}

void RunVoronoiFT2D(float *sp2, int height, int width, int depth, double *vol)
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
	g.stack = (node *)malloc(g.max_size * sizeof(node));

	#pragma omp for
	for (k = 0; k < depth; k++)
	{
		int j;
		for (j = 0; j < width; j++)
		{
			int i;
			for (i = 0; i < height; i++)
			{
				if (vol[k * slice_stride + i * width + j] != -1.0)
				{
					int fv_k = int(vol[k * slice_stride + i * width + j]) \
							/ slice_stride;
					
					int fv_i = (int(vol[k * slice_stride + i * width + j]) \
							% slice_stride) / width;
					
					int fv_j = (int(vol[k * slice_stride + i * width + j]) \
							% slice_stride) % width;
					
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

						while (g.size >= 2 && removeFT2D(sp2, &g, w, Rd))
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
			for (i = 0; i < height; i++)
			{
				double minDist = DBL_MAX;
				int minIndex = -1;

				int ite = 0;
				while(ite < g.size)
				{
					double tempDist = ED(sp2, i, j, k, &(g.stack[ite]));

					if(tempDist < minDist)
					{
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] = \
					double(g.stack[minIndex].fv_pos[2] * slice_stride + \
					g.stack[minIndex].fv_pos[0] * width + \
					g.stack[minIndex].fv_pos[1]);
			}
			g.size = 0;
		}
	}
	free(g.stack);
	}
}

void RunVoronoiFT3D(float *sp2, int height, int width, int depth, double *vol)
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
	g.stack = (node *)malloc(g.max_size * sizeof(node));

	#pragma omp for
	for (i = 0; i < height; i++)
	{
		int j;
		for (j = 0; j < width; j++)
		{
			int k;
			for (k = 0; k < depth; k++)
			{
				if (vol[k * slice_stride + i * width + j] != -1.0)
				{
					int fv_k = int(vol[k * slice_stride + i * width + j]) \
							/ slice_stride;

					int fv_i = (int(vol[k * slice_stride + i * width + j]) \
							% slice_stride) / width;

					int fv_j = (int(vol[k * slice_stride + i * width + j]) \
							% slice_stride) % width;
					
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

						while (g.size >= 2 && removeFT3D(sp2, &g, w, Rd))
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
			for (k = 0; k < depth; k++)
			{
				double minDist = DBL_MAX;
				int minIndex = -1;

				int ite = 0;
				while(ite < g.size)
				{
					double tempDist = ED(sp2, i, j, k, &(g.stack[ite]));

					if(tempDist < minDist)
					{
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] = \
					double(g.stack[minIndex].fv_pos[2] * slice_stride + \
					g.stack[minIndex].fv_pos[0] * width + \
					g.stack[minIndex].fv_pos[1]);
			}
			g.size = 0;
		}
	}
	free(g.stack);
	}
}

int removeFT2D(float *sp2, GNodes *g, int *w, int *Rd)
{
	node u = g->stack[g->size - 2];
	node v = g->stack[g->size - 1];

	double a = (v.fv_pos[0] - u.fv_pos[0]) * sqrt(sp2[0]);
	double b = (w[0] - v.fv_pos[0]) * sqrt(sp2[0]);
	double c = a + b;

	double vRd = 0.0;
	double uRd = 0.0;
	double wRd = 0.0;

	int i = 1;
	for (i; i < 3; i++)
	{
		vRd += (v.fv_pos[i] - Rd[i]) * (v.fv_pos[i] - Rd[i]) * sp2[i];
		uRd += (u.fv_pos[i] - Rd[i]) * (u.fv_pos[i] - Rd[i]) * sp2[i];
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]) * sp2[i];
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0.0);

}

int removeFT3D(float *sp2, GNodes *g, int *w, int *Rd)
{
	node u = g->stack[g->size - 2];
	node v = g->stack[g->size - 1];

	double a = (v.fv_pos[2] - u.fv_pos[2]) * sqrt(sp2[2]);
	double b = (w[2] - v.fv_pos[2]) * sqrt(sp2[2]);
	double c = a + b;

	double vRd = 0;
	double uRd = 0;
	double wRd = 0;

	int i = 0;
	for (i; i < 2; i++)
	{
		vRd += (v.fv_pos[i] - Rd[i]) * (v.fv_pos[i] - Rd[i]) * sp2[i];
		uRd += (u.fv_pos[i] - Rd[i]) * (u.fv_pos[i] - Rd[i]) * sp2[i];
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]) * sp2[i];
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0.0);
}

double ED(float *sp2, \
	int vol_i, int vol_j, int vol_k, \
	node *fv)
{
	double temp = 0;

	temp = (fv->fv_pos[0] - vol_i) * (fv->fv_pos[0] - vol_i) * sp2[0] +\
			(fv->fv_pos[1] - vol_j) * (fv->fv_pos[1] - vol_j) * sp2[1] +\
			(fv->fv_pos[2] - vol_k) * (fv->fv_pos[2] - vol_k) * sp2[2];

	return sqrt(temp);
}
