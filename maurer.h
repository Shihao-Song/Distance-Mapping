#ifndef _MAURER_H_
#define _MAURER_H_

#include <math.h>
#include <float.h>

#include "volume.h"
#include "gnodes.h"

void maurerFT(uchar *, int, int, int, int *);
void VoronoiFT(int, uchar *, int, int, int, int *);
void RunVoronoiFT1D(uchar *, int, int, int, int *);
void RunVoronoiFT2D(int, int, int, int *);
void RunVoronoiFT3D(int, int, int, int *);

int removeFT2D(GNodes *g, int *w, int *Rd);
int removeFT3D(GNodes *g, int *w, int *Rd);
// Calculate Euclidean Distance
double ED(int, int, int, node *);

#endif