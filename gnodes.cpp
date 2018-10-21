#include "gnodes.h"

void push(GNodes *g, int fv_i, int fv_j, int fv_k)
{
	if (g == NULL)
	{
		printf("GNodes Operation Failed. Error: NULL Pointer. \n");
		return;
	}

	if (g->size == g->max_size)
	{
		printf("GNodes Operation Failed. Error: Stack is Full. \n");
		return;
	}
	else
	{
		g->stack[g->size].fv_pos[0] = fv_i;
		g->stack[g->size].fv_pos[1] = fv_j;
		g->stack[g->size].fv_pos[2] = fv_k;

		g->size++;
	}
}

void pop(GNodes *g)
{
	if (g == NULL)
	{
		printf("GNodes Operation Failed. Error: NULL Pointer. \n");
		return;
	}

	if (g->size == 0)
	{
		printf("GNodes Operation Failed. Error: Stack is Empty. \n");
		return;
	}
	else
	{
		g->size--;
	}
}