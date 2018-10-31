#ifndef _MAURER_H_
#define _MAURER_H_

#include <cmath>
#include <cfloat>
#include <deque>
#include <vector>

#include "omp.h"

using namespace std;

/*
	maurerFT: takes the raw volume and generates its FT (Feature Transformation)
*/
void maurerFT(unsigned char *, float *, int, int, int, double *);

/*
	Helper functions, the following functions are the building blocks of maurerFT()
*/
void voronoiFT(int, unsigned char *, float *, int, int, int, double *);

void runVoronoiFT1D(unsigned char *, float *, int, int, int, double *);
void runVoronoiFT2D(float *, int, int, int, double *);
void runVoronoiFT3D(float *, int, int, int, double *);

int removeFT2D(float *, deque<vector<int>> &g_nodes, int *w, int *Rd);
int removeFT3D(float *, deque<vector<int>> &g_nodes, int *w, int *Rd);

double ED(float *, int, int, int, vector<int> &fv);// Calculate Euclidean Distance

#endif
