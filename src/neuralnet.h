#include "dataprocessing.h"

int Training();
double **FeedForwarding(const ENTITY *,const NET_DEFINE *,double **,double **, double **, double **);
void AssignDimension(int *, int, int, int);
double **Relu(double **, int);
double **Sigmoid(double **, int);
double **Softmax(double **, int);
int BackPropagation(double **, double **, double **, double **, double **,const ENTITY *,const NET_DEFINE *);
int EvaluateAccuracy(ENTITY *,const NET_DEFINE *,double **,double **, double **, double **);
int Validation(const ENTITY *,const NET_DEFINE *,double **,double **, double **, double **, double *);
double MeanSquareError(const ENTITY *, double **, int);
