#ifndef _EXHAUSTIVE_D_M_H_
#define _EXHAUSTIVE_D_M_H_

#include <cfloat>
#include <cstdlib>

#include "../tmp/Vol.h"

#include "../boundary_face_dist/boundary_face_distance.h"

using namespace std;

class Exhaustive_Distance_Map
{
public:
	Exhaustive_Distance_Map(char *mapping_scheme)
	{
		boundary_face_dist_calc = new Boundary_Face_Distance(mapping_scheme);
	}

	~Exhaustive_Distance_Map()
	{
		free(boundary_face_dist_calc);
	}

	void run_exhaustive(Vol *vol);

	double *dist_mapping_exhaustive;

protected:

	Boundary_Face_Distance *boundary_face_dist_calc;

	void exhaustiveFT(unsigned char *vol, float *sp2,
                		int height, int width, int depth,
                		double *output);

	void exhaustiveSearch(unsigned char *vol, float *sp2,
                		int vol_i, int vol_j, int vol_k,
                		int height, int width, int depth,
                		double *output);
};

#endif
