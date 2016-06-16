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

int Training()
{
	int i,j;
	double base = 0.1;
	ENTITY *entityPtr = NULL;
	srand(time(NULL));

	if(strcmp(netDefinition.weightAssignment, "Zero") == 0)
		base = 0;	

	//memory allocation for weights between layers
	i2hWeights = (double**)malloc(sizeof(double*)*netDefinition.inputLayerNeuronNum);
	for(i = 0; i < netDefinition.inputLayerNeuronNum; ++i){
		i2hWeights[i] = (double*)malloc(sizeof(double)*netDefinition.hiddenLayerNeuronNum);
		for(j = 0; j < netDefinition.hiddenLayerNeuronNum; ++j)
			i2hWeights[i][j] = ((rand() % 5) + 1) * base;
	}
	h2oWeights = (double**)malloc(sizeof(double*)*netDefinition.hiddenLayerNeuronNum);
	for(i = 0; i < netDefinition.hiddenLayerNeuronNum; ++i){
		h2oWeights[i] = (double*)malloc(sizeof(double)*netDefinition.outputLayerNeuronNum);
		for(j = 0; j < netDefinition.outputLayerNeuronNum; ++j)
			h2oWeights[i][j] = ((rand() % 5) + 1) * base;
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
	
	//weigths
	PrintMatrix(i2hWeights,netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);
	PrintMatrix(h2oWeights,netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);	
	for(i = 0; i < netDefinition.epoch; ++i){
		if((i % netDefinition.validationCycle) == 0)	
			EvaluateAccuracy(validationSet, &netDefinition, i2hWeights, h2oWeights, i2hBias, h2oBias);
		entityPtr = trainingSet;
		while(entityPtr != NULL){
			FeedForwarding(entityPtr, &netDefinition, i2hWeights, h2oWeights, i2hBias, h2oBias);
			entityPtr = entityPtr->pNext;
		}
	}
	//weigths
	PrintMatrix(i2hWeights,netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);
	PrintMatrix(h2oWeights,netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);	

	return 0;
}

double **FeedForwarding(const ENTITY *entity,const NET_DEFINE *netDef,double **i2hWeights,double **h2oWeights, double **i2oBias, double **h2oBias)
{
	double **inputLayer;
	double **hiddenLayer;
	double **outputLayer;
	double **(*activation)(double **,int);
	
	//malloc and assign the initial value	
	inputLayer = (double**)malloc(sizeof(double*)*1);
	for(int i = 0; i < 1; ++i){
		inputLayer[i] = (double*)malloc(sizeof(double)*netDef->inputLayerNeuronNum);
	 	for(int j = 0; j < netDef->inputLayerNeuronNum; ++j)
			inputLayer[i][j] = *((entity->attributes)+j);
	}

	//asssign the function pointer
	if(strcmp(netDef->activationFunction, "Sigmoid") == 0)
		activation = Sigmoid;
	else
		activation = Relu;

	//matrix operation (i 2 h)
	hiddenLayer = M_Multiply(inputLayer, i2hWeights, 1, netDef->inputLayerNeuronNum, netDef->hiddenLayerNeuronNum);
	hiddenLayer = M_Add(hiddenLayer, i2hBias, 1, netDef->hiddenLayerNeuronNum);
	hiddenLayer = activation(hiddenLayer, netDef->hiddenLayerNeuronNum);
	//matrix operation (h 2 o)
	outputLayer = M_Multiply(hiddenLayer, h2oWeights, 1, netDef->hiddenLayerNeuronNum, netDef->outputLayerNeuronNum);
	outputLayer = M_Add(outputLayer, h2oBias, 1, netDef->outputLayerNeuronNum);
	outputLayer = activation(outputLayer, netDef->outputLayerNeuronNum);
	
	BackPropagation(outputLayer, hiddenLayer, inputLayer, i2hWeights, h2oWeights, entity, netDef);

	Free2DMemory(inputLayer,1);
	Free2DMemory(hiddenLayer,1);
	Free2DMemory(outputLayer,1);

	return NULL;		
}

int BackPropagation(double **output, double **hidden, double **input, double **i2hWeights, double **h2oWeights, const ENTITY *entity,const NET_DEFINE *netDef)
{
	double **errorO = (double **)malloc(sizeof(double *));	
	errorO[0] = (double *)malloc(sizeof(double) * netDef->outputLayerNeuronNum);
	double **errorH = (double **)malloc(sizeof(double *));
	errorH[0] = (double *)malloc(sizeof(double) * netDef->hiddenLayerNeuronNum);
	double **i2hUpdate = NULL;	
	double **h2oUpdate = NULL;
	double **transpose;
	
	//calculate error between output layer and hidden layer
	for(int i = 0; i < netDef->outputLayerNeuronNum; ++i){
		errorO[0][i] = output[0][i] * (1 - output[0][i]) * ((entity->catagory)[i]-output[0][i]);
	}
	
	transpose = M_Transpose(hidden, 1, netDef->hiddenLayerNeuronNum);
	h2oUpdate = M_Multiply(transpose, errorO, netDef->hiddenLayerNeuronNum, 1, netDef->outputLayerNeuronNum);
	h2oUpdate = Multiply(h2oUpdate, netDef->learningRate, netDef->hiddenLayerNeuronNum, netDef->outputLayerNeuronNum);
	h2oWeights = M_Add(h2oWeights, h2oUpdate, netDef->hiddenLayerNeuronNum, netDef->outputLayerNeuronNum);
	Free2DMemory(transpose, netDef->hiddenLayerNeuronNum);
	
	//calculate error between hidden layer and output layer
	for(int i = 0; i < netDef->hiddenLayerNeuronNum; ++i){
		errorH[0][i] = 0;
		for(int j = 0; j < netDef->outputLayerNeuronNum; ++j){
			errorH[0][i] += (h2oWeights[i][j] * errorO[0][j]);
		}
		errorH[0][i] = errorH[0][i] * (hidden[0][i] * (1 - hidden[0][i]));
	}	
	transpose = M_Transpose(input, 1, netDef->inputLayerNeuronNum);
	i2hUpdate = M_Multiply(transpose, errorH, netDef->inputLayerNeuronNum, 1, netDef->hiddenLayerNeuronNum);
	i2hUpdate = Multiply(i2hUpdate, netDef->learningRate, netDef->inputLayerNeuronNum, netDef->hiddenLayerNeuronNum);
	i2hWeights = M_Add(i2hWeights, i2hUpdate, netDef->inputLayerNeuronNum, netDef->hiddenLayerNeuronNum);
	Free2DMemory(transpose, netDef->inputLayerNeuronNum);
	
	free(errorO);
	free(errorH);
	Free2DMemory(i2hUpdate, netDef->inputLayerNeuronNum);
	Free2DMemory(h2oUpdate, netDef->hiddenLayerNeuronNum);
	return 0;
}

int EvaluateAccuracy(ENTITY *entity,const NET_DEFINE *netDef,double **i2hWeights,double **h2oWeights, double **i2hBias, double **h2oBias)
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
		if(Validation(entityPtr, netDef, i2hWeights, h2oWeights, i2hBias, h2oBias, &meanSquareError) == 1)
			++count;
		entityPtr = entityPtr->pNext;
		++total;
	}
	printf("Accuracy : %0.2f\t",((count*100)/total) * 0.01);
	printf("Loss : %lf\n", meanSquareError);
	cycle += netDef->validationCycle;
	return 0;
} 

int Validation(const ENTITY *entity,const NET_DEFINE *netDef,double **i2hWeights,double **h2oWeights, double **i2hBias, double **h2oBias, double *meanSquareError)
{
	double **inputLayer;
	double **hiddenLayer;
	double **outputLayer;
	double **(*activation)(double **,int);
	
	//malloc and assign the initial value	
	inputLayer = (double**)malloc(sizeof(double*)*1);
	for(int i = 0; i < 1; ++i){
		inputLayer[i] = (double*)malloc(sizeof(double)*netDef->inputLayerNeuronNum);
		for(int j = 0; j < netDef->inputLayerNeuronNum; ++j){
			inputLayer[i][j] = (entity->attributes)[j];
		}
	}

	//asssign the function pointer
	if(strcmp(netDef->activationFunction, "Sigmoid") == 0)
		activation = Sigmoid;
	else
		activation = Relu;

	//matrix operation
	hiddenLayer = M_Multiply(inputLayer, i2hWeights, 1, netDef->inputLayerNeuronNum, netDef->hiddenLayerNeuronNum);
	hiddenLayer = M_Add(hiddenLayer, i2hBias, 1, netDef->hiddenLayerNeuronNum);
	hiddenLayer = activation(hiddenLayer, netDef->hiddenLayerNeuronNum);
	outputLayer = M_Multiply(hiddenLayer, h2oWeights, 1, netDef->hiddenLayerNeuronNum, netDef->outputLayerNeuronNum);
	outputLayer = M_Add(outputLayer, h2oBias, 1, netDef->outputLayerNeuronNum);
	outputLayer = activation(outputLayer, netDef->outputLayerNeuronNum);
	
	//calculate mean square error
	*meanSquareError = MeanSquareError(entity, outputLayer, netDef->outputLayerNeuronNum);

	//softmax
	outputLayer = Softmax(outputLayer, netDef->outputLayerNeuronNum);
	
	int maxIndex = 0;
	for(int i = 0; i < netDef->outputLayerNeuronNum; ++i){	
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
