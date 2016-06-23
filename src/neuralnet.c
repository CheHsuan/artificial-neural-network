#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
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

static inline __forceinline
double **Allocate2DMemory(int row, int column)
{
	double **ptr = (double **)malloc(sizeof(double*) * row);
	for(int i = 0; i < row; ++i){
		ptr[i] = (double *)malloc(sizeof(double) * column);
		for(int j = 0; j < column; ++j)
			ptr[i][j] = 0; 
	}
	return ptr;
}

static inline __forceinline
void Free2DMemory(double **matrix, int row)
{
	for(int i = 0; i < row; ++i){
		free(matrix[i]);		
	}	
	free(matrix);
}

int Training()
{
	double base = 0.1;
	ENTITY *entityPtr = NULL;
	WEIGHTS *updateWeights = (WEIGHTS *)malloc(sizeof(WEIGHTS) * SYS_CORE);
	srand(time(NULL));

	if(strcmp(netDefinition.weightAssignment, "Zero") == 0)
		base = 0;
	//memory allocation and initialization for weights between layers
	i2hWeights = Allocate2DMemory(netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);
	for(int i = 0; i < netDefinition.inputLayerNeuronNum; ++i){
		for(int j = 0; j < netDefinition.hiddenLayerNeuronNum; ++j)
			i2hWeights[i][j] = ((rand() % 10) - 5) * base;
	}
	h2oWeights = Allocate2DMemory(netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);
	for(int i = 0; i < netDefinition.hiddenLayerNeuronNum; ++i){
		for(int j = 0; j < netDefinition.outputLayerNeuronNum; ++j)
			h2oWeights[i][j] = ((rand() % 10) - 5) * base;
	}
	//memory allocation and initialization for bias
	i2hBias = Allocate2DMemory(1, netDefinition.hiddenLayerNeuronNum);
	for(int j = 0; j < netDefinition.hiddenLayerNeuronNum; ++j)
		i2hBias[0][j] = 1;
	h2oBias = Allocate2DMemory(1, netDefinition.outputLayerNeuronNum);
	for(int j = 0; j < netDefinition.outputLayerNeuronNum; ++j)
		h2oBias[0][j] = 1;
	//memory allocation and initialization for weight increments
	for(int i = 0; i < SYS_CORE; ++i){
		updateWeights[i].i2hWeights = Allocate2DMemory(netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);
		updateWeights[i].h2oWeights = Allocate2DMemory(netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);
		for(int row = 0; row < netDefinition.inputLayerNeuronNum; ++row){
			for(int column = 0; column < netDefinition.hiddenLayerNeuronNum; ++column)
				updateWeights[i].i2hWeights[row][column] = 0;
		}
		for(int row = 0; row < netDefinition.hiddenLayerNeuronNum; ++row){
			for(int column = 0; column < netDefinition.outputLayerNeuronNum; ++column)
				updateWeights[i].h2oWeights[row][column] = 0;
		}
	}
	pthread_t *threads;
	threads = (pthread_t *)malloc(sizeof(pthread_t) * SYS_CORE);
	THREADARG arg;
	int threadcount;
	for(int i = 0; i < netDefinition.epoch; ++i){
		if((i % netDefinition.validationCycle) == 0)	
			EvaluateAccuracy(validationSet);
		entityPtr = trainingSet;
		while(entityPtr != NULL){
			for(threadcount = -1; threadcount++ < SYS_CORE-1;entityPtr = entityPtr->pNext){
				if(entityPtr != NULL){
					arg.entity = entityPtr;
					arg.update = updateWeights+threadcount;
					pthread_create(&threads[threadcount], NULL, (void *)FeedForwarding,(void *)&arg);	
				}
				else 
					break;
			}
			for(int j = 0; j < threadcount; ++j)
				pthread_join(threads[j],NULL);
			ParameterServer(updateWeights);
		}
	}

	//free updates
	for(int i = 0; i < SYS_CORE; ++i){
		Free2DMemory(updateWeights[i].i2hWeights, netDefinition.inputLayerNeuronNum);
		Free2DMemory(updateWeights[i].h2oWeights, netDefinition.hiddenLayerNeuronNum);
	}
	free(updateWeights);


	return 0;
}

void ParameterServer(WEIGHTS *updates)
{
	for(int i = 0; i < SYS_CORE; ++i){
		i2hWeights = M_Add(i2hWeights, updates[i].i2hWeights, netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);	
		h2oWeights = M_Add(h2oWeights, updates[i].h2oWeights, netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);
		for(int row = 0; row < netDefinition.inputLayerNeuronNum; ++row){
			for(int column = 0; column < netDefinition.hiddenLayerNeuronNum; ++column)
				updates[i].i2hWeights[row][column] = 0;
		}
		for(int row = 0; row < netDefinition.hiddenLayerNeuronNum; ++row){
			for(int column = 0; column < netDefinition.outputLayerNeuronNum; ++column)
				updates[i].h2oWeights[row][column] = 0;
		}
	}
}

void FeedForwarding(void *threadArg)
{
	THREADARG *arg = (THREADARG *)threadArg;
	ENTITY *entity = arg->entity;
	WEIGHTS *update = arg->update;
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
	
	BackPropagation(outputLayer, hiddenLayer, inputLayer, entity, update);

	Free2DMemory(inputLayer,1);
	Free2DMemory(hiddenLayer,1);
	Free2DMemory(outputLayer,1);		
}

void  BackPropagation(double **output, double **hidden, double **input, const ENTITY *entity, WEIGHTS *update)
{
	double **errorO = Allocate2DMemory(1, netDefinition.outputLayerNeuronNum);	
	double **errorH = Allocate2DMemory(1, netDefinition.hiddenLayerNeuronNum);
	double **transpose = NULL;

	//calculate error between output layer and hidden layer
	for(int i = 0; i < netDefinition.outputLayerNeuronNum; ++i){
		errorO[0][i] = output[0][i] * (1 - output[0][i]) * ((entity->catagory)[i]-output[0][i]);
	}
	
	transpose = Allocate2DMemory(netDefinition.hiddenLayerNeuronNum, 1);
       	transpose = M_Transpose(hidden, transpose, 1, netDefinition.hiddenLayerNeuronNum);
	update->h2oWeights = M_Multiply(transpose, errorO, update->h2oWeights, netDefinition.hiddenLayerNeuronNum, 1, netDefinition.outputLayerNeuronNum);
	update->h2oWeights = Multiply(update->h2oWeights, netDefinition.learningRate, netDefinition.hiddenLayerNeuronNum, netDefinition.outputLayerNeuronNum);
	Free2DMemory(transpose, netDefinition.hiddenLayerNeuronNum);
	
	//calculate error between hidden layer and output layer
	for(int i = 0; i < netDefinition.hiddenLayerNeuronNum; ++i){
		for(int j = 0; j < netDefinition.outputLayerNeuronNum; ++j){
			errorH[0][i] += (h2oWeights[i][j] * errorO[0][j]);
		}
		errorH[0][i] = errorH[0][i] * (hidden[0][i] * (1 - hidden[0][i]));
	}
	transpose = Allocate2DMemory(netDefinition.inputLayerNeuronNum, 1);	
	transpose = M_Transpose(input, transpose, 1, netDefinition.inputLayerNeuronNum);
	update->i2hWeights = M_Multiply(transpose, errorH, update->i2hWeights, netDefinition.inputLayerNeuronNum, 1, netDefinition.hiddenLayerNeuronNum);
	update->i2hWeights = Multiply(update->i2hWeights, netDefinition.learningRate, netDefinition.inputLayerNeuronNum, netDefinition.hiddenLayerNeuronNum);
	Free2DMemory(transpose, netDefinition.inputLayerNeuronNum);
	
	free(errorO);
	free(errorH);
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

void FreeMemory()
{
	//free weights and bias
	Free2DMemory(i2hWeights, netDefinition.inputLayerNeuronNum);
	Free2DMemory(h2oWeights, netDefinition.hiddenLayerNeuronNum);
	Free2DMemory(i2hBias, 1);
	Free2DMemory(h2oBias, 1);

	FreeDataMemory();
}
