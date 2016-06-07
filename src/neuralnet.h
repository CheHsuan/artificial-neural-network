#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "xmlparser.h"
#include "matrix.h"

typedef struct NET_DEFINE{
	double learningRate;
	int epoch;
	int inputLayerNeuronNum;
	int hiddenLayerNeuronNum;
	int outputLayerNeuronNum;
	char *activationFunction;
	char *weightAssignment;
	int validationCycle;	
}NET_DEFINE;

typedef struct ENTITY{
	double *attributes;
	double *catagory;
	struct ENTITY *pNext;
}ENTITY;

int LoadNetDefinition(char *);
int ReadNetDefinition(NET_DEFINE *,char *);
int LoadTrainingSet(char *);
int LoadValidationSet(char *);
int LoadTestingSet(char *);
ENTITY *ReadDataSet(ENTITY *, const NET_DEFINE *, char *);
ENTITY *Add2List(char *, const NET_DEFINE *, ENTITY *);
int Training();
double **FeedForwarding(const ENTITY *,const NET_DEFINE *,double **,double **, double **);
void AssignDimension(int *, int, int, int);
double **Relu(double **, int);
double **Sigmoid(double **, int);
double **Softmax(double **, int);
int BackPropagation(double **, double **, double **, double **, double **,const ENTITY *,const NET_DEFINE *);
int EvaluateAccuracy(ENTITY *,const NET_DEFINE *,double **,double **, double **);
int Validation(const ENTITY *,const NET_DEFINE *,double **,double **, double **);
void Test();
void Free2DMemory(double **, int);
