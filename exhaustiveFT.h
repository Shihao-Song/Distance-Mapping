#ifndef _COMPUTE_GOLD_H_
#define _COMPUTE_GOLD_H_

#include "volume.h"
#include <limits.h>

// exhaustiveFT - Closet Feature Transform using exhaustive
// search
void exhaustiveFT(uchar *, int, int, int, int *);
void exhaustiveSearch(int, int, int, uchar *, \
	int, int, int, int *);

#endif