#ifndef _COMPUTE_GOLD_H_
#define _COMPUTE_GOLD_H_

#include <float.h>

typedef unsigned char uchar;

void exhaustiveFT(uchar *, float *, int, int, int, double *);
void exhaustiveSearch(uchar *, float *, \
		int, int, int, \
		int, int, int, \
		double *);

#endif
