#include "matrix.h"

double **M_Multiply(double **a, double **b, int *dimension)
{
	double **c = (double **)malloc(sizeof(double *)*dimension[0]);
	for(int i = 0; i < dimension[0]; ++i)
		c[i] = (double *)malloc(sizeof(double)*dimension[2]);

	for(int i = 0; i < dimension[0]; ++i){
		for(int j = 0; j < dimension[2]; ++j){
			c[i][j] = 0;
			for(int k = 0; k < dimension[1]; ++k){
				c[i][j] = c[i][j] + a[i][k] * b[k][j];	
			}
//			printf("%lf ", c[i][j]);
		}
//		printf("\n");	
	}	
	return c;
}

double **M_Add(double **a, double **b, int row, int column)
{
	for(int i = 0; i < row; ++i){
		for(int j = 0; j < column; ++j){
			a[i][j] = a[i][j] + b[i][j];
//			printf("%lf ", a[i][j]);
		}
//		printf("\n");
	}	
	return a;
}
