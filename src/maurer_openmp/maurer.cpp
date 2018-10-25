#include "maurer.h"

void maurerFT(uchar *vol, int height, int width, int depth, \
	int *output)
{
	int dim;

	for (dim = 1; dim <= 3; dim++)
	{
		VoronoiFT(dim, vol, height, width, depth, output);
	}
}

void VoronoiFT(int dim, 
	uchar *vol, int height, int width, int depth, int *output)
{
	switch (dim)
	{
		case 1:
			RunVoronoiFT1D(vol, height, width, depth, output);
			break;
		case 2:
			RunVoronoiFT2D(height, width, depth, output);
			break;
		case 3:
			RunVoronoiFT3D(height, width, depth, output);
			break;
		default:
			break;
	}
}

void RunVoronoiFT1D(uchar *vol, int height, int width, int depth,\
	int *output)
{
	GNodes g;
	g.size = 0;
	g.max_size = width;
	g.stack = (node *)malloc(g.max_size * sizeof(node));

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
					(ED(i, j, k, &(g.stack[ite])) > \
					ED(i, j, k, &(g.stack[ite+1]))))
				{
					ite++;
				}

				output[k * slice_stride + i * width + j] = \
					g.stack[ite].fv_pos[2] * slice_stride + \
					g.stack[ite].fv_pos[0] * width + \
					g.stack[ite].fv_pos[1];
			}
			g.size = 0;
		}
	}
	free(g.stack);
}

void RunVoronoiFT2D(int height, int width, int depth, int *vol)
{
	GNodes g;
	g.size = 0;
	g.max_size = height;
	g.stack = (node *)malloc(g.max_size * sizeof(node));

	int Rd[3];
	int w[3];

	// Distance between slices
	int slice_stride = height * width;

	int i, j, k;
	for (k = 0; k < depth; k++)
	{
		for (j = 0; j < width; j++)
		{
			for (i = 0; i < height; i++)
			{
				if (vol[k * slice_stride + i * width + j] != -1)
				{
					int fv_k = vol[k * slice_stride + i * width + j] \
							/ slice_stride;
					
					int fv_i = (vol[k * slice_stride + i * width + j] \
							% slice_stride) / width;
					
					int fv_j = (vol[k * slice_stride + i * width + j] \
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

						while (g.size >= 2 && removeFT2D(&g, w, Rd))
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
					double tempDist = ED(i, j, k, &(g.stack[ite]));

					if(tempDist < minDist)
					{
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] = \
					g.stack[minIndex].fv_pos[2] * slice_stride + \
					g.stack[minIndex].fv_pos[0] * width + \
					g.stack[minIndex].fv_pos[1];
			}
			g.size = 0;
		}
	}
	free(g.stack);
}

void RunVoronoiFT3D(int height, int width, int depth, int *vol)
{
	GNodes g;
	g.size = 0;
	g.max_size = depth;
	g.stack = (node *)malloc(g.max_size * sizeof(node));

	int Rd[3];
	int w[3];

	// Distance between slices
	int slice_stride = height * width;

	int i, j, k;
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			for (k = 0; k < depth; k++)
			{
				if (vol[k * slice_stride + i * width + j] != -1)
				{
					int fv_k = vol[k * slice_stride + i * width + j] \
							/ slice_stride;

					int fv_i = (vol[k * slice_stride + i * width + j] \
							% slice_stride) / width;

					int fv_j = (vol[k * slice_stride + i * width + j] \
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

						while (g.size >= 2 && removeFT3D(&g, w, Rd))
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
					double tempDist = ED(i, j, k, &(g.stack[ite]));

					if(tempDist < minDist)
					{
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] = \
					g.stack[minIndex].fv_pos[2] * slice_stride + \
					g.stack[minIndex].fv_pos[0] * width + \
					g.stack[minIndex].fv_pos[1];
			}
			g.size = 0;
		}
	}
	free(g.stack);
}

int removeFT2D(GNodes *g, int *w, int *Rd)
{
	node u = g->stack[g->size - 2];
	node v = g->stack[g->size - 1];

	int a = v.fv_pos[0] - u.fv_pos[0];
	int b = w[0] - v.fv_pos[0];
	int c = a + b;

	int vRd = 0;
	int uRd = 0;
	int wRd = 0;

	int i = 1;
	for (i; i < 3; i++)
	{
		vRd += (v.fv_pos[i] - Rd[i]) * (v.fv_pos[i] - Rd[i]);
		uRd += (u.fv_pos[i] - Rd[i]) * (u.fv_pos[i] - Rd[i]);
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]);
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0);

}

int removeFT3D(GNodes *g, int *w, int *Rd)
{
	node u = g->stack[g->size - 2];
	node v = g->stack[g->size - 1];

	int a = v.fv_pos[2] - u.fv_pos[2];
	int b = w[2] - v.fv_pos[2];
	int c = a + b;

	int vRd = 0;
	int uRd = 0;
	int wRd = 0;

	int i = 0;
	for (i; i < 2; i++)
	{
		vRd += (v.fv_pos[i] - Rd[i]) * (v.fv_pos[i] - Rd[i]);
		uRd += (u.fv_pos[i] - Rd[i]) * (u.fv_pos[i] - Rd[i]);
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]);
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0);
}

double ED(int vol_i, int vol_j, int vol_k, node *fv)
{
	int temp = 0;

	temp = (fv->fv_pos[0] - vol_i) * (fv->fv_pos[0] - vol_i) +\
			(fv->fv_pos[1] - vol_j) * (fv->fv_pos[1] - vol_j) +\
			(fv->fv_pos[2] - vol_k) * (fv->fv_pos[2] - vol_k);

	return sqrt((double)temp);
}
