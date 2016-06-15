#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "xmlparser.h"
#include "matrix.h"
#include "neuralnet.h"

NET_DEFINE netDefinition;
ENTITY *trainingSet = NULL;
ENTITY *validationSet = NULL;
ENTITY *testingSet = NULL;
double **i2hWeights = NULL;
double **h2oWeights = NULL;
double **i2hBias = NULL;
double **h2oBias = NULL;

static void PrintMatrix(double **matrix, int row, int column)
{
	for(int i = 0; i < row; ++i){
		for(int j = 0; j < column; ++j){
			printf("%lf ", matrix[i][j]);
		}
		printf("\n");
	}	
}

int LoadNetDefinition(char *srcFile)
{
	printf("Load the network definition file......\n");
	if(ReadNetDefinition(&netDefinition, srcFile) == 0){
		printf("Done!\n");
		return 0;
	}
	printf("Error in reading the file (%s)!\n",srcFile);
	exit(0);
}

int ReadNetDefinition(NET_DEFINE *netDefinition, char *srcFile)
{
	int size = 1000;
        char *xml =(char*)malloc(sizeof(char)*(size+1));
	char learningRate[5];
	char epoch[5];  
	char inputLayerNeuronNum[5];
 	char hiddenLayerNeuronNum[5];
     	char outputLayerNeuronNum[5];
	char cycle[5];
	
	//parse the network definition file	
	if(FileToStr(xml, srcFile, &size) == -1)
		return -1;
	if(InnerText(learningRate, xml, "<LearningRate>", "</LearningRate>") == NULL){
		printf("The xml file doesn't contain the <LearningRate> or </LearningRate> label.");
		return -1;
	}
	if(InnerText(epoch, xml, "<Epoch>", "</Epoch>") == NULL){
		printf("The xml file doesn't contain the <Epoch> or </Epoch> label.");
		return -1;
	}
	if(InnerText(inputLayerNeuronNum, xml, "<InputLayerNeuronNum>", "</InputLayerNeuronNum>") == NULL){
		printf("The xml file doesn't contain the <InputLayerNeuronNum> or </InputLayerNeuronNum> label.");
		return -1;
	}
	if(InnerText(hiddenLayerNeuronNum, xml, "<HiddenLayerNeuronNum>", "</HiddenLayerNeuronNum>") == NULL){
		printf("The xml file doesn't contain the <HiddenLayerNeuronNum> or </HiddenLayerNeuronNum> label.");
		return -1;
	}
	if(InnerText(outputLayerNeuronNum, xml, "<OutputLayerNeuronNum>", "</OutputLayerNeuronNum>") == NULL){
		printf("The xml file doesn't contain the <OutputLayerNeuronNum> or </OutputLayerNeuronNum> label.");
		return -1;
	}
	netDefinition->activationFunction = (char *)malloc(sizeof(char)*20);
	if(InnerText(netDefinition->activationFunction, xml, "<ActivationFunction>", "</ActivationFunction>") == NULL){
		printf("The xml file doesn't contain the <ActivationFunction> or </ActivationFunction> label.");
		free(netDefinition->activationFunction);
		return -1;
	}
	netDefinition->weightAssignment = (char *)malloc(sizeof(char)*20);
	if(InnerText(netDefinition->weightAssignment, xml, "<WeightAssignment>", "</WeightAssignment>") == NULL){
		printf("The xml file doesn't contain the <WeightAssignment> or </WeightAssignment> label.");
		free(netDefinition->weightAssignment);
		return -1;
	}
	if(InnerText(cycle, xml, "<ValidationCycle>", "</ValidationCycle>") == NULL){
		printf("The xml file doesn't contain the <ValidationCycle> or </ValidationCycle> label.");
		return -1;
	}
	netDefinition->learningRate = atof(learningRate);
	netDefinition->epoch = atoi(epoch);
	netDefinition->inputLayerNeuronNum = atoi(inputLayerNeuronNum);
	netDefinition->hiddenLayerNeuronNum = atoi(hiddenLayerNeuronNum);
	netDefinition->outputLayerNeuronNum = atoi(outputLayerNeuronNum);
	netDefinition->validationCycle = atoi(cycle);
	
	return 0; 
}

int LoadTrainingSet(char *srcFile)
{
	printf("Load the training data set......\n");
	if((trainingSet = ReadDataSet(trainingSet, &netDefinition, srcFile)) != NULL){
		printf("Done!\n");
		return 0;
	}
	printf("Error in reading the file (%s)!\n",srcFile);
	exit(0);
}

int LoadValidationSet(char *srcFile)
{
	printf("Load the validation data set......\n");
	if((validationSet = ReadDataSet(validationSet, &netDefinition, srcFile)) != NULL){
		printf("Done!\n");
		return 0;
	}
	printf("Error in reading the file (%s)!\n",srcFile);
	exit(0);
}

int LoadTestingSet(char *srcFile)
{
	printf("Load the testing data set......\n");
	if((testingSet = ReadDataSet(testingSet, &netDefinition, srcFile)) != NULL){
		printf("Done!\n");
		return 0;
	}
	printf("Error in reading the file (%s)!\n",srcFile);
	exit(0);
}

ENTITY *ReadDataSet(ENTITY *dataSet,const NET_DEFINE *netDef, char *srcFile)
{
	FILE *fp = NULL;
	char buffer[200];
	
	ENTITY *entityListTail = (ENTITY *)malloc(sizeof(ENTITY));
	dataSet = entityListTail;

	if((fp = fopen(srcFile, "r")) == NULL){
		printf("File not found!\n");
		assert(fp);
		return NULL;
	}

	//read the file and parse it
	while(fgets(buffer, sizeof(buffer), fp) != NULL){
		entityListTail = Add2List(buffer, netDef, entityListTail);
	}
	entityListTail = dataSet;
	dataSet	= dataSet->pNext;
	//free unused memory
	free(entityListTail);
	fclose(fp);	

	return dataSet;		
}

ENTITY *Add2List(char *buffer, const NET_DEFINE *netDef, ENTITY *entityListTail)
{
	ENTITY *entity = NULL;
	char *token;
	int i = 0;

	entity = (ENTITY*)malloc(sizeof(ENTITY));
	entity->attributes = (double*)malloc(sizeof(double)*netDef->inputLayerNeuronNum);
	entity->catagory = (double*)malloc(sizeof(double)*netDef->outputLayerNeuronNum);
	//parse the string to get the class and the column values
	token = strtok(buffer, " ,\t");
	for(i = 0; i < netDef->outputLayerNeuronNum; ++i){
		if( i == atoi(token))
			*((entity->catagory)+i) = 1;
		else
			*((entity->catagory)+i) = 0;
	}
	i = 0;
	/*TODO modify the expression*/
	while((token = strtok(NULL, " ,\t")) != NULL){
		*((entity->attributes)+(i++)) = atof(token);
	}
	if(i != netDef->inputLayerNeuronNum){
		printf("The dataset format is incompatible to network definition.\n");
		assert(i == netDef->inputLayerNeuronNum);
		free(entity->attributes);
		free(entity);
		return NULL;
	}

	entity->pNext = NULL;
	entityListTail->pNext = entity;
	entityListTail = entity;

	return entityListTail;	
}

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
	int dimension[3];
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
	AssignDimension(dimension, 1, netDef->inputLayerNeuronNum, netDef->hiddenLayerNeuronNum);
	hiddenLayer = M_Multiply(inputLayer, i2hWeights, dimension);
	hiddenLayer = M_Add(hiddenLayer, i2hBias, 1, netDef->hiddenLayerNeuronNum);
	hiddenLayer = activation(hiddenLayer, netDef->hiddenLayerNeuronNum);
	//matrix operation (h 2 o)
	AssignDimension(dimension, 1, netDef->hiddenLayerNeuronNum, netDef->outputLayerNeuronNum);
	outputLayer = M_Multiply(hiddenLayer, h2oWeights, dimension);
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
	double *errorO = (double *)malloc(sizeof(double)*netDef->outputLayerNeuronNum);
	double *errorH = (double *)malloc(sizeof(double)*netDef->hiddenLayerNeuronNum);
	
	//calculate error between output layer and hidden layer
	for(int i = 0; i < netDef->outputLayerNeuronNum; ++i){
		errorO[i] = output[0][i] * (1 - output[0][i]) * ((entity->catagory)[i]-output[0][i]);
	}
	for(int i = 0; i < netDef->hiddenLayerNeuronNum; ++i){
		for(int j = 0; j < netDef->outputLayerNeuronNum; ++j){
			h2oWeights[i][j] += (netDef->learningRate * hidden[0][i] * errorO[j]);
		}
	}
	//calculate error between hidden layer and output layer
	for(int i = 0; i < netDef->hiddenLayerNeuronNum; ++i){
		errorH[i] = 0;
		for(int j = 0; j < netDef->outputLayerNeuronNum; ++j){
			errorH[i] += (h2oWeights[i][j] * errorO[j]);
		}
		errorH[i] = errorH[i] * (hidden[0][i] * (1 - hidden[0][i]));
	}	
	for(int i = 0; i < netDef->inputLayerNeuronNum; ++i){
		for(int j = 0; j < netDef->hiddenLayerNeuronNum; ++j){
			i2hWeights[i][j] += (netDef->learningRate * input[0][i] * errorH[j]);
		}
	}
	free(errorO);
	free(errorH);
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
	int dimension[3];
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
	AssignDimension(dimension, 1, netDef->inputLayerNeuronNum, netDef->hiddenLayerNeuronNum);
	hiddenLayer = M_Multiply(inputLayer, i2hWeights, dimension);
	hiddenLayer = M_Add(hiddenLayer, i2hBias, 1, netDef->hiddenLayerNeuronNum);
	hiddenLayer = activation(hiddenLayer, netDef->hiddenLayerNeuronNum);
	AssignDimension(dimension, 1, netDef->hiddenLayerNeuronNum, netDef->outputLayerNeuronNum);
	outputLayer = M_Multiply(hiddenLayer, h2oWeights, dimension);
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

void Free2DMemory(double **matrix, int row)
{
	for(int i = 0; i < row; ++i){
		free(matrix[i]);		
	}	
	free(matrix);
}

void FreeDataList(ENTITY *ptr)
{
	ENTITY *tmp;
	while(ptr != NULL){
		tmp = ptr;
		free(ptr->attributes);
		free(ptr->catagory);
		ptr = ptr->pNext;
		free(tmp);
	}
}

void FreeMemory()
{
	//free weights and bias
	Free2DMemory(i2hWeights, netDefinition.inputLayerNeuronNum);
	Free2DMemory(h2oWeights, netDefinition.hiddenLayerNeuronNum);
	Free2DMemory(i2hBias, 1);
	Free2DMemory(h2oBias, 1);
	
	//free training data set, validation data set and testing data set
	FreeDataList(trainingSet);
	FreeDataList(validationSet);
	FreeDataList(testingSet);

	//free memory in netDefinition
	free(netDefinition.activationFunction);
	free(netDefinition.weightAssignment);
}
