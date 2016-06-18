#include "dataprocessing.h"

typedef struct WEIGHTS{
	double **i2hWeights;
	double **h2oWeights;
}WEIGHTS;

int Training();
double **FeedForwarding(const ENTITY *);
double **Relu(double **, int);
double **Sigmoid(double **, int);
double **Softmax(double **, int);
int BackPropagation(double **, double **, double **, const ENTITY*);
int EvaluateAccuracy(ENTITY *);
int Validation(const ENTITY *, double *);
double MeanSquareError(const ENTITY *, double **, int);
