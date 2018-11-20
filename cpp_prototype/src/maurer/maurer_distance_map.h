#ifndef _MAURER_FT_H_
#define _MAURER_FT_H_

#include <cstdlib>
#include <cmath>
#include <cfloat>

#include <iostream>

#include "omp.h"

#include "../tmp/Vol.h"

using namespace std;

class Maurer_Distance_Map
{
public:
	Maurer_Distance_Map () {}
	~Maurer_Distance_Map () {}

	void run_maurer(Vol *vol);

protected:
	/*
		In-class data structure	
	*/
	typedef struct
	{
		int fv_pos[3];
	}node;

	typedef struct
	{
		unsigned int size;

		unsigned int max_size;

		node* nodes;
	}GNodes;

	void push(GNodes *g, int fv_i, int fv_j, int fv_k)
	{
		g->nodes[g->size].fv_pos[0] = fv_i;
		g->nodes[g->size].fv_pos[1] = fv_j;
		g->nodes[g->size].fv_pos[2] = fv_k;

		g->size++;
	}

	void pop(GNodes *g)
	{
		g->size--;
	}

	/*
		Maurer Distance Mappings
	*/
	void maurerFT(unsigned char *vol,
        		float *sp2,
        		int height, int width, int depth,
        		double *output);
	
	void voronoiFT(int dim,
        		unsigned char *vol,
        		float *sp2,
        		int height, int width, int depth,
        		double *output);

	void runVoronoiFT1D(unsigned char *vol, 
				float *sp2,
                		int height, int width, int depth,
                		double *output);

	void runVoronoiFT2D(float *sp2,
                		int height, int width, int depth,
                		double *vol);

	void runVoronoiFT3D(float *sp2, 
				int height, int width, int depth, 
				double *vol);

	int removeFT2D(float *sp2, 
			GNodes *g_nodes, int *w, int *Rd);

	int removeFT3D(float *sp2, 
			GNodes *g_nodes, int *w, int *Rd);

	// Euclidean Distance
	double ED(float *sp2,
                	int vol_i, int vol_j, int vol_k,
                	node *fv);
};

#endif

