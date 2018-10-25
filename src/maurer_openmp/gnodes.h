#ifndef _GNODES_H_
#define _GNODES_H_

#include <stdio.h>
#include <stdlib.h>

/*
	GNodes is a (stack) data structure that is used
	in VoronoiFT (maurer's FT).
*/

typedef struct
{
	int fv_pos[3];
}node;

typedef struct
{
	// Keep track of the current size of stack
	unsigned int size;

	// Maximum size of the stack
	unsigned int max_size;

	node* stack;

}GNodes;

/*
	Stack (GNodes) Operation Functions Defined Here
*/
void push(GNodes *, int, int, int);
void pop(GNodes *);

#endif
