#include <stdio.h>

static inline __forceinline
double **M_Multiply(double **a, double **b, double **c, int x, int y, int z)
{
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

static inline __forceinline
double **Multiply(double **a, double rational, int row, int column)
{
	for(int i = 0; i < row; ++i)
		for(int j = 0; j < column; ++j)
			a[i][j] *= rational; 
	return a;
}

static inline __forceinline
double **M_Add(double **a, double **b, int row, int column)
{
	for(int i = 0; i < row; ++i){
		for(int j = 0; j < column; ++j){
			a[i][j] = a[i][j] + b[i][j];
		}
	}	
	return a;
}

static inline __forceinline
double **Add(double **a, double rational, int row, int column)
{
	for(int i = 0; i < row; ++i)
		for(int j = 0; j < column; ++j)
			a[i][j] += rational;
	return a;
}

static inline __forceinline
double **M_Transpose(double **a, double **b, int row, int column)
{
	for(int i = 0; i < column; ++i){
		for(int j = 0; j < row; ++j){
			b[i][j] = a[j][i];
		}
	}
	return b;
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
