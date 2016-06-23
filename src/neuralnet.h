#include "dataprocessing.h"

typedef struct WEIGHTS{
	double **i2hWeights;
	double **h2oWeights;
}WEIGHTS;

typedef struct THREADARG{
	ENTITY *entity;
	WEIGHTS *update;
}THREADARG;

int Training();
void ParameterServer(WEIGHTS *);
void FeedForwarding(void *);
double **Relu(double **, int);
double **Sigmoid(double **, int);
double **Softmax(double **, int);
void BackPropagation(double **, double **, double **, const ENTITY*, WEIGHTS *);
int EvaluateAccuracy(ENTITY *);
int Validation(const ENTITY *, double *);
double MeanSquareError(const ENTITY *, double **, int);
void FreeMemory();
