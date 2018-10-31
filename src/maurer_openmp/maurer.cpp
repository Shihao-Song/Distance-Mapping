#include "maurer.h"

void maurerFT(unsigned char *vol, 
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

void voronoiFT(int dim,
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

void runVoronoiFT1D(unsigned char *vol, float *sp2,
		int height, int width, int depth,
		double *output)
{
	// GNodes
	deque<vector<int>> g_nodes;

	// Distance between slices
	int slice_stride = height * width;

	int k;

	#pragma omp parallel shared(vol, sp2, output) private(g_nodes,k)
	{
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
					vector<int> fv {i, j, k};
					
					g_nodes.push_back(fv);
				}
			}

			if (g_nodes.size() == 0)
			{
				continue;
			}

			// Query partial voronoi diagram
			for (j = 0; j < width; j++)
			{
				int ite = 0;
				while ((ite < (g_nodes.size() - 1)) &&
					(ED(sp2, i, j, k, g_nodes[ite]) >
					ED(sp2, i, j, k, g_nodes[ite+1])))
				{
					ite++;
				}

				output[k * slice_stride + i * width + j] =
					double(g_nodes[ite][2] * slice_stride +
					g_nodes[ite][0] * width +
					g_nodes[ite][1]);
			}
			g_nodes.clear();
		}
	}
	}
}

void runVoronoiFT2D(float *sp2, 
		int height, int width, int depth, 
		double *vol)
{
	deque<vector<int>> g_nodes;
	
	int Rd[3];
	int w[3];

	// Distance between slices
	int slice_stride = height * width;

	int k;
	#pragma omp parallel shared(sp2, vol) private(g_nodes, Rd, w, k)
    	{
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
					int fv_k = int(vol[k * slice_stride + i * width + j])
							/ slice_stride;
					
					int fv_i = (int(vol[k * slice_stride + i * width + j])
							% slice_stride) / width;
					
					int fv_j = (int(vol[k * slice_stride + i * width + j])
							% slice_stride) % width;
					
					if(g_nodes.size() < 2)
					{
						vector<int> fv {fv_i, fv_j, fv_k};

						g_nodes.push_back(fv);
					}
					else
					{
						w[0] = fv_i;
						w[1] = fv_j;
						w[2] = fv_k;

						Rd[0] = i;
						Rd[1] = j;
						Rd[2] = k;

						while (g_nodes.size() >= 2 && 
							removeFT2D(sp2, g_nodes, w, Rd))
						{
							g_nodes.pop_back();
						}

						vector<int> fv {fv_i, fv_j, fv_k};

						g_nodes.push_back(fv);

					}
				}
			}

			if (g_nodes.size() == 0)
			{
				continue;
			}

			// Query partial voronoi diagram
			for (i = 0; i < height; i++)
			{
				double minDist = DBL_MAX;
				int minIndex = -1;

				int ite = 0;
				while(ite < g_nodes.size())
				{
					double tempDist = ED(sp2, i, j, k, g_nodes[ite]);

					if(tempDist < minDist)
					{
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] = \
					double(g_nodes[minIndex][2] * slice_stride +
					g_nodes[minIndex][0] * width +
					g_nodes[minIndex][1]);
			}
			g_nodes.clear();
		}
	}
	}
}

void runVoronoiFT3D(float *sp2, int height, int width, int depth, double *vol)
{
	deque<vector<int>> g_nodes;

	int Rd[3];
	int w[3];

	// Distance between slices
	int slice_stride = height * width;

	int i;
	#pragma omp parallel shared(sp2, vol) private(g_nodes, Rd, w, i)
    	{
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
					int fv_k = int(vol[k * slice_stride + i * width + j])
							/ slice_stride;

					int fv_i = (int(vol[k * slice_stride + i * width + j])
							% slice_stride) / width;

					int fv_j = (int(vol[k * slice_stride + i * width + j])
							% slice_stride) % width;
					
					if(g_nodes.size() < 2)
					{
						vector<int> fv {fv_i, fv_j, fv_k};

						g_nodes.push_back(fv);
					}
					else
					{
						w[0] = fv_i;
						w[1] = fv_j;
						w[2] = fv_k;

						Rd[0] = i;
						Rd[1] = j;
						Rd[2] = k;

						while (g_nodes.size() >= 2 && 
							removeFT3D(sp2, g_nodes, w, Rd))
						{
							g_nodes.pop_back();
						}

						vector<int> fv {fv_i, fv_j, fv_k};

						g_nodes.push_back(fv);

					}
				}
			}
				
			if (g_nodes.size() == 0)
			{
				continue;
			}

			// Query partial voronoi diagram
			for (k = 0; k < depth; k++)
			{
				double minDist = DBL_MAX;
				int minIndex = -1;

				int ite = 0;
				while(ite < g_nodes.size())
				{
					double tempDist = ED(sp2, i, j, k, g_nodes[ite]);

					if(tempDist < minDist)
					{
						minDist = tempDist;
						minIndex = ite;
					}
					ite++;
				}

				vol[k * slice_stride + i * width + j] = \
					double(g_nodes[minIndex][2] * slice_stride + \
					g_nodes[minIndex][0] * width + \
					g_nodes[minIndex][1]);
			}
			g_nodes.clear();
		}
	}
	}
}

int removeFT2D(float *sp2, deque<vector<int>> &g_nodes, int *w, int *Rd)
{
	vector<int> u = g_nodes[g_nodes.size() - 2];
	vector<int> v = g_nodes[g_nodes.size() - 1];

	double a = (v[0] - u[0]) * sqrt(sp2[0]);
	double b = (w[0] - v[0]) * sqrt(sp2[0]);
	double c = a + b;

	double vRd = 0.0;
	double uRd = 0.0;
	double wRd = 0.0;

	int i = 1;
	for (i; i < 3; i++)
	{
		vRd += (v[i] - Rd[i]) * (v[i] - Rd[i]) * sp2[i];
		uRd += (u[i] - Rd[i]) * (u[i] - Rd[i]) * sp2[i];
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]) * sp2[i];
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0.0);

}

int removeFT3D(float *sp2, deque<vector<int>> &g_nodes, int *w, int *Rd)
{
	vector<int> u = g_nodes[g_nodes.size() - 2];
	vector<int> v = g_nodes[g_nodes.size() - 1];

	double a = (v[2] - u[2]) * sqrt(sp2[2]);
	double b = (w[2] - v[2]) * sqrt(sp2[2]);
	double c = a + b;

	double vRd = 0;
	double uRd = 0;
	double wRd = 0;

	int i = 0;
	for (i; i < 2; i++)
	{
		vRd += (v[i] - Rd[i]) * (v[i] - Rd[i]) * sp2[i];
		uRd += (u[i] - Rd[i]) * (u[i] - Rd[i]) * sp2[i];
		wRd += (w[i] - Rd[i]) * (w[i] - Rd[i]) * sp2[i];
	}

	return (c * vRd - b * uRd - a * wRd - a * b * c > 0.0);
}

double ED(float *sp2, 
		int vol_i, int vol_j, int vol_k, 
		vector<int> &fv)
{
	double temp = 0;

        temp = (fv[0] - vol_i) * (fv[0] - vol_i) * sp2[0] +\
                        (fv[1] - vol_j) * (fv[1] - vol_j) * sp2[1] +\
                        (fv[2] - vol_k) * (fv[2] - vol_k) * sp2[2];

        return sqrt(temp);
}
