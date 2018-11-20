#ifndef _BOUNDARY_FACE_DISTANCE_H_
#define _BOUNDARY_FACE_DISTANCE_H_

#include <cmath>
#include <cfloat>
#include <string>

#include "omp.h"

#include "../tmp/Vol.h"

using namespace std;

class Boundary_Face_Distance
{
public:
	Boundary_Face_Distance(char * mapping_scheme) : mapping_scheme(string(mapping_scheme)) {}
	~Boundary_Face_Distance() {}

	void run_boundary_face_distance_calculation(Vol *vol, double *distance_mapping);

protected:
	string mapping_scheme;

	void boundaryFaceDist(int scheme,
                        	unsigned char *raw_vol,
		        	float *sp2,
                        	int height, int width, int depth,
				double *ed_out);
	
	double distToClosetFacePointOfCFV(
				float *sp2,
				int i, int j, int k,
                                int cfv_i, int cfv_j, int cfv_k,
				unsigned char cfv_val);

	double distToFacePoint(int unchanged,
				float *sp2,
                        	double i, double j, double k,
				double cfv_i, double cfv_j, double cfv_k);

	double calcDist(
			float *sp2,
			double i, double j, double k,
			double target_i, double target_j, double target_k);

	bool isKthBitSet(int n, int k);

};

#endif
