#ifndef _MAURER_H_
#define _MAURER_H_

#include <math.h>
#include <float.h>

#include "volume.h"
#include "gnodes.h"

/*
	maurerFT: takes the raw volume and generates its FT (Feature Transformation)
*/
void maurerFT(uchar *, int, int, int, int *);

/*
	Helper functions, the following functions are the building blocks of maurerFT()
*/
void VoronoiFT(int, uchar *, int, int, int, int *);
void RunVoronoiFT1D(uchar *, int, int, int, int *);
void RunVoronoiFT2D(int, int, int, int *);
void RunVoronoiFT3D(int, int, int, int *);

int removeFT2D(GNodes *g, int *w, int *Rd);
int removeFT3D(GNodes *g, int *w, int *Rd);

double ED(int, int, int, node *); // Calculate Euclidean Distance

#endif
