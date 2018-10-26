#ifndef _MAURER_H_
#define _MAURER_H_

#include <math.h>
#include <float.h>

#include "omp.h"

#include "gnodes.h"

typedef unsigned char uchar;

/*
	maurerFT: takes the raw volume and generates its FT (Feature Transformation)
*/
void maurerFT(uchar *, float *, int, int, int, double *);

/*
	Helper functions, the following functions are the building blocks of maurerFT()
*/
void VoronoiFT(int, uchar *, float *, int, int, int, double *);

void RunVoronoiFT1D(uchar *, float *, int, int, int, double *);
void RunVoronoiFT2D(float *, int, int, int, double *);
void RunVoronoiFT3D(float *, int, int, int, double *);

int removeFT2D(float *, GNodes *g, int *w, int *Rd);
int removeFT3D(float *, GNodes *g, int *w, int *Rd);

double ED(float *, int, int, int, node *); // Calculate Euclidean Distance

#endif
