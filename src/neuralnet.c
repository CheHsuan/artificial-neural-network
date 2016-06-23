#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "xmlparser.h"
#include "matrix.h"
#include "neuralnet.h"

extern NET_DEFINE netDefinition;
extern ENTITY *trainingSet;
extern ENTITY *validationSet;
extern ENTITY *testingSet;
double **i2hWeights = NULL;
double **h2oWeights = NULL;
double **i2hBias = NULL;
double **h2oBias = NULL;
WEIGHTS *updates;

int Training()
{
	int i,j;
	double base = 0.1;
	ENTITY *entityPtr = NULL;
	updates = (WEIGHTS *)malloc(sizeof(WEIGHTS) * SYS_CORE);
	srand(time(NULL));

	if(strcmp(netDefinition.weightAssignment, "Zero") == 0)
		base = 0;	

	//memory allocation for weights between layers
	i2hWeights = (double**)malloc(sizeof(double*)*netDefinition.inputLayerNeuronNum);
	for(i = 0; i < netDefinition.inputLayerNeuronNum; ++i){
		i2hWeights[i] = (double*)malloc(sizeof(double)*netDefinition.hiddenLayerNeuronNum);
		for(j = 0; j < netDefinition.hiddenLayerNeuronNum; ++j)
			i2hWeights[i][j] = ((rand() % 10) - 5) * base;
	}
	h2oWeights = (double**)malloc(sizeof(double*)*netDefinition.hiddenLayerNeuronNum);
	for(i = 0; i < netDefinition.hiddenLayerNeuronNum; ++i){
		h2oWeights[i] = (double*)malloc(sizeof(double)*netDefinition.outputLayerNeuronNum);
		for(j = 0; j < netDefinition.outputLayerNeuronNum; ++j)
			h2oWeights[i][j] = ((rand() % 10) - 5) * base;
	}
	//memory allocation for bias
	i2hBias = (double**)malloc(sizeof(double*));
	i2hBias[0] = (double*)malloc(sizeof(double)*netDefinition.hiddenLayerNeuronNum);
		for(j = 0; j < netDefinition.hiddenLayerNeuronNum; ++j)
			i2hBias[0][j] = 1;
	h2oBias = (double**)malloc(sizeof(double*));
	h2oBias[0] = (double*)malloc(sizeof(double)*netDefinition.outputLayerNeuronNum);
		for(j = 0; j < netDefinition.outputLayerNeuronNum; ++j)
			h2oBias[0][j] = 1;
	
	for(i = 0; i < netDefinition.epoch; ++i){
		if((i % netDefinition.validationCycle) == 0)	
			EvaluateAccuracy(validationSet);
		entityPtr = trainingSet;
		while(entityPtr != NULL){
			FeedForwarding(entityPtr);
			entityPtr = entityPtr->pNext;
		}
	}
	return 0;
}

void ParameterServer(WEIGHTS *updates)
{
	for(int i = 0; i < SYS_CORE; ++i){
		i2hWeights = M_Add(i2hWeights, updates[i].i2hWeights, netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);	
		h2oWeights = M_Add(h2oWeights, updates[i].h2oWeights, netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);
	}	
}

double **FeedForwarding(const ENTITY *entity)
{
	double **inputLayer;
	double **hiddenLayer;
	double **outputLayer;
	double **(*activation)(double **,int);
	
	//malloc and assign the initial value	
	inputLayer = (double**)malloc(sizeof(double*) * 1);
	inputLayer[0] = (double*)malloc(sizeof(double)*netDefinition.inputLayerNeuronNum);
	for(int j = 0; j < netDefinition.inputLayerNeuronNum; ++j)
		inputLayer[0][j] = *((entity->attributes)+j);
	hiddenLayer = (double**)malloc(sizeof(double*) * 1);
	hiddenLayer[0] = (double*)malloc(sizeof(double)*netDefinition.hiddenLayerNeuronNum);
	for(int j = 0; j < netDefinition.hiddenLayerNeuronNum; ++j)
		hiddenLayer[0][j] = 0;
	outputLayer = (double**)malloc(sizeof(double*) * 1);
	outputLayer[0] = (double*)malloc(sizeof(double)*netDefinition.outputLayerNeuronNum);
	for(int j = 0; j < netDefinition.outputLayerNeuronNum; ++j)
		outputLayer[0][j] = 0;	

	//asssign the function pointer
	if(strcmp(netDefinition.activationFunction, "Sigmoid") == 0)
		activation = Sigmoid;
	else
		activation = Relu;

	//matrix operation (i 2 h)
	hiddenLayer = M_Multiply(inputLayer, i2hWeights, hiddenLayer, 1, netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);
	hiddenLayer = M_Add(hiddenLayer, i2hBias, 1, netDefinition.hiddenLayerNeuronNum);
	hiddenLayer = activation(hiddenLayer, netDefinition.hiddenLayerNeuronNum);
	//matrix operation (h 2 o)
	outputLayer = M_Multiply(hiddenLayer, h2oWeights, outputLayer, 1, netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);
	outputLayer = M_Add(outputLayer, h2oBias, 1, netDefinition.outputLayerNeuronNum);
	outputLayer = activation(outputLayer, netDefinition.outputLayerNeuronNum);
	
	BackPropagation(outputLayer, hiddenLayer, inputLayer, entity);

	Free2DMemory(inputLayer,1);
	Free2DMemory(hiddenLayer,1);
	Free2DMemory(outputLayer,1);

	return NULL;		
}

int BackPropagation(double **output, double **hidden, double **input, const ENTITY *entity)
{
	double **errorO = (double **)malloc(sizeof(double *));	
	errorO[0] = (double *)malloc(sizeof(double) * netDefinition.outputLayerNeuronNum);
	double **errorH = (double **)malloc(sizeof(double *));
	errorH[0] = (double *)malloc(sizeof(double) * netDefinition.hiddenLayerNeuronNum);
	double **i2hUpdate = NULL;	
	double **h2oUpdate = NULL;
	double **transpose;
	//memory allocation for weights between layers
	i2hUpdate = (double**)malloc(sizeof(double*)*netDefinition.inputLayerNeuronNum);
	for(int i = 0; i < netDefinition.inputLayerNeuronNum; ++i){
		i2hUpdate[i] = (double*)malloc(sizeof(double)*netDefinition.hiddenLayerNeuronNum);
		for(int j = 0; j < netDefinition.hiddenLayerNeuronNum; ++j)
			i2hUpdate[i][j] = 0;
	}
	h2oUpdate = (double**)malloc(sizeof(double*)*netDefinition.hiddenLayerNeuronNum);
	for(int i = 0; i < netDefinition.hiddenLayerNeuronNum; ++i){
		h2oUpdate[i] = (double*)malloc(sizeof(double)*netDefinition.outputLayerNeuronNum);
		for(int j = 0; j < netDefinition.outputLayerNeuronNum; ++j)
			h2oUpdate[i][j] = 0;
	}

	//calculate error between output layer and hidden layer
	for(int i = 0; i < netDefinition.outputLayerNeuronNum; ++i){
		errorO[0][i] = output[0][i] * (1 - output[0][i]) * ((entity->catagory)[i]-output[0][i]);
	}
	
        transpose = (double **)malloc(sizeof(double *) * netDefinition.hiddenLayerNeuronNum);
	for(int i = 0; i < netDefinition.hiddenLayerNeuronNum; ++i)
		transpose[i] = (double *)malloc(sizeof(double) * 1);
	transpose = M_Transpose(hidden, transpose, 1, netDefinition.hiddenLayerNeuronNum);
	h2oUpdate = M_Multiply(transpose, errorO, h2oUpdate, netDefinition.hiddenLayerNeuronNum, 1, netDefinition.outputLayerNeuronNum);
	h2oUpdate = Multiply(h2oUpdate, netDefinition.learningRate, netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);
	h2oWeights = M_Add(h2oWeights, h2oUpdate, netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);
	Free2DMemory(transpose, netDefinition.hiddenLayerNeuronNum);
	
	//calculate error between hidden layer and output layer
	for(int i = 0; i < netDefinition.hiddenLayerNeuronNum; ++i){
		errorH[0][i] = 0;
		for(int j = 0; j < netDefinition.outputLayerNeuronNum; ++j){
			errorH[0][i] += (h2oWeights[i][j] * errorO[0][j]);
		}
		errorH[0][i] = errorH[0][i] * (hidden[0][i] * (1 - hidden[0][i]));
	}	
	transpose = (double **)malloc(sizeof(double *) * netDefinition.inputLayerNeuronNum);
	for(int i = 0; i < netDefinition.inputLayerNeuronNum; ++i)
		transpose[i] = (double *)malloc(sizeof(double) * 1);
	transpose = M_Transpose(input, transpose, 1, netDefinition.inputLayerNeuronNum);
	i2hUpdate = M_Multiply(transpose, errorH, i2hUpdate, netDefinition.inputLayerNeuronNum, 1, netDefinition.hiddenLayerNeuronNum);
	i2hUpdate = Multiply(i2hUpdate, netDefinition.learningRate, netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);
	i2hWeights = M_Add(i2hWeights, i2hUpdate, netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);
	Free2DMemory(transpose, netDefinition.inputLayerNeuronNum);
	
	free(errorO);
	free(errorH);
	Free2DMemory(i2hUpdate, netDefinition.inputLayerNeuronNum);
	Free2DMemory(h2oUpdate, netDefinition.hiddenLayerNeuronNum);
	return 0;
}

int EvaluateAccuracy(ENTITY *entity)
{
	int total = 0;
	int count = 0;
	static int cycle = 0;
	ENTITY *entityPtr = entity;
	double meanSquareError;
	time_t timep; 
	struct tm *p; 
	time(&timep); 
	p = localtime(&timep); 
	printf("[%d] %d:%d:%d\t", cycle, p->tm_hour, p->tm_min, p->tm_sec);
	while(entityPtr != NULL){
		if(Validation(entityPtr, &meanSquareError) == 1)
			++count;
		entityPtr = entityPtr->pNext;
		++total;
	}
	printf("Accuracy : %0.2f\t",((count*100)/total) * 0.01);
	printf("Loss : %lf\n", meanSquareError);
	cycle += netDefinition.validationCycle;
	return 0;
} 

int Validation(const ENTITY *entity, double *meanSquareError)
{
	double **inputLayer;
	double **hiddenLayer;
	double **outputLayer;
	double **(*activation)(double **,int);
	
	//malloc and assign the initial value	
	inputLayer = (double**)malloc(sizeof(double*) * 1);
	inputLayer[0] = (double*)malloc(sizeof(double)*netDefinition.inputLayerNeuronNum);
	for(int j = 0; j < netDefinition.inputLayerNeuronNum; ++j)
		inputLayer[0][j] = *((entity->attributes)+j);
	hiddenLayer = (double**)malloc(sizeof(double*) * 1);
	hiddenLayer[0] = (double*)malloc(sizeof(double)*netDefinition.hiddenLayerNeuronNum);
	for(int j = 0; j < netDefinition.hiddenLayerNeuronNum; ++j)
		hiddenLayer[0][j] = 0;
	outputLayer = (double**)malloc(sizeof(double*) * 1);
	outputLayer[0] = (double*)malloc(sizeof(double)*netDefinition.outputLayerNeuronNum);
	for(int j = 0; j < netDefinition.outputLayerNeuronNum; ++j)
		outputLayer[0][j] = 0;	

	//asssign the function pointer
	if(strcmp(netDefinition.activationFunction, "Sigmoid") == 0)
		activation = Sigmoid;
	else
		activation = Relu;

	//matrix operation
	hiddenLayer = M_Multiply(inputLayer, i2hWeights, hiddenLayer, 1, netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);
	hiddenLayer = M_Add(hiddenLayer, i2hBias, 1, netDefinition.hiddenLayerNeuronNum);
	hiddenLayer = activation(hiddenLayer, netDefinition.hiddenLayerNeuronNum);
	outputLayer = M_Multiply(hiddenLayer, h2oWeights, outputLayer, 1, netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);
	outputLayer = M_Add(outputLayer, h2oBias, 1, netDefinition.outputLayerNeuronNum);
	outputLayer = activation(outputLayer, netDefinition.outputLayerNeuronNum);
	
	//calculate mean square error
	*meanSquareError = MeanSquareError(entity, outputLayer, netDefinition.outputLayerNeuronNum);

	//softmax
	outputLayer = Softmax(outputLayer, netDefinition.outputLayerNeuronNum);
	
	int maxIndex = 0;
	for(int i = 0; i < netDefinition.outputLayerNeuronNum; ++i){	
		if(outputLayer[0][i] > outputLayer[0][maxIndex])
			maxIndex = i;
	}
	Free2DMemory(inputLayer,1);
	Free2DMemory(hiddenLayer,1);
	Free2DMemory(outputLayer,1);
	
	return ((entity->catagory)[maxIndex] == 1 ? 1 : 0);		
}

void AssignDimension(int *a, int b, int c, int d)
{
	a[0] = b;
	a[1] = c;
	a[2] = d;
}

double **Relu(double **a, int column)
{
	for(int i = 0; i < column; ++i)
		if(a[0][i] < 0)
			a[0][i]	= 0;
	return a;
}

double **Sigmoid(double **a, int column)
{
	for(int i = 0; i < column; ++i){
		a[0][i]	= 1 / (1 + exp(0-a[0][i]));
	}
	return a;
}

double **Softmax(double **a, int column)
{
	double denominator = 0;
	for(int i = 0; i < column; ++i){
		denominator += exp(a[0][i]);
	}
	for(int i = 0; i < column; ++i){
		a[0][i] = exp(a[0][i]) / denominator;
	}	
	return a;
}

double MeanSquareError(const ENTITY *entity, double **output, int column)
{
	double mse = 0;
	for(int i = 0; i < column; ++i){
		mse += (((entity->catagory)[i]-output[0][i]) * ((entity->catagory)[i]-output[0][i]));
	}
	return 0.5*mse;
}
