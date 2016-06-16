#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

double **M_Multiply(double **a, double **b, int x, int y, int z)
{
	double **c = (double **)malloc(sizeof(double *) * x);
	for(int i = 0; i < x; ++i)
		c[i] = (double *)malloc(sizeof(double) * z);

	for(int i = 0; i < x; ++i){
		for(int j = 0; j < z; ++j){
			c[i][j] = 0;
			for(int k = 0; k < y; ++k){
				c[i][j] = c[i][j] + a[i][k] * b[k][j];	
			}
		}
	}	
	return c;
}

double **Multiply(double **a, double rational, int row, int column)
{
	for(int i = 0; i < row; ++i)
		for(int j = 0; j < column; ++j)
			a[i][j] *= rational; 
	return a;
}

double **M_Add(double **a, double **b, int row, int column)
{
	for(int i = 0; i < row; ++i){
		for(int j = 0; j < column; ++j){
			a[i][j] = a[i][j] + b[i][j];
		}
	}	
	return a;
}

double **Add(double **a, double rational, int row, int column)
{
	for(int i = 0; i < row; ++i)
		for(int j = 0; j < column; ++j)
			a[i][j] += rational;
	return a;
}

double **M_Transpose(double **a, int row, int column)
{
	double **transpose = (double **)malloc(sizeof(double *) * column);
	for(int i = 0; i < column; ++i){
		transpose[i] = (double *)malloc(sizeof(double) * row);
		for(int j = 0; j < row; ++j){
			transpose[i][j] = a[j][i];
		}
	}
	return transpose;
}

void PrintMatrix(double **matrix, int row, int column)
{
	for(int i = 0; i < row; ++i){
		for(int j = 0; j < column; ++j){
			printf("%lf ", matrix[i][j]);
		}
		printf("\n");
	}	
}
