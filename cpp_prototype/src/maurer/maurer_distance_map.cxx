#include "maurer_distance_map.h"

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


