#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "xmlparser.h"
#include "dataprocessing.h"

NET_DEFINE netDefinition;
ENTITY *trainingSet = NULL;
ENTITY *validationSet = NULL;
ENTITY *testingSet = NULL;
ENTITY **dividedListPtr;
extern double **i2hWeights;
extern double **h2oWeights;
extern double **i2hBias;
extern double **h2oBias;

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
	printf("CPU cores : %d\n", SYS_CORE);
	int dataSetSize = 0;
	printf("Load the training data set......\n");
	if((trainingSet = ReadDataSet(trainingSet, &netDefinition, srcFile, &dataSetSize)) != NULL){
		printf("Done!\n");
		dividedListPtr = DivideDataSet(trainingSet,dataSetSize);
		return 0;
	}
	printf("Error in reading the file (%s)!\n",srcFile);
	exit(0);
}

int LoadValidationSet(char *srcFile)
{
	int dataSetSize = 0;
	printf("Load the validation data set......\n");
	if((validationSet = ReadDataSet(validationSet, &netDefinition, srcFile, &dataSetSize)) != NULL){
		printf("Done!\n");
		return 0;
	}
	printf("Error in reading the file (%s)!\n",srcFile);
	exit(0);
}

int LoadTestingSet(char *srcFile)
{
	int dataSetSize = 0;
	printf("Load the testing data set......\n");
	if((testingSet = ReadDataSet(testingSet, &netDefinition, srcFile, &dataSetSize)) != NULL){
		printf("Done!\n");
		return 0;
	}
	printf("Error in reading the file (%s)!\n",srcFile);
	exit(0);
}

ENTITY *ReadDataSet(ENTITY *dataSet,const NET_DEFINE *netDef, char *srcFile , int *dataSetSize)
{
	FILE *fp = NULL;
	char buffer[10000];
	int count = 0;
	
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
		count++;
	}
	*dataSetSize = count;
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
			entity->catagory[i] = 1;
		else
			entity->catagory[i] = 0;
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
		free(entity->catagory);
		free(entity);
		return NULL;
	}

	entity->pNext = NULL;
	entityListTail->pNext = entity;
	entityListTail = entity;

	return entityListTail;	
}

ENTITY **DivideDataSet(ENTITY *head, int entityCount)
{
	int subsetSize;
	int count = 0;
	int num = 0;
	ENTITY **dataPtr;
	ENTITY *entity = head;
	
	subsetSize = entityCount / SYS_CORE;
	dataPtr = (ENTITY **)malloc(sizeof(ENTITY *) * (SYS_CORE + 1));
	dataPtr[count++] = entity;
	while(entity != NULL){
		if(num == subsetSize){
			dataPtr[count++] = entity;
			num = 0;
		}
		else{
			num += 1;
		}
		entity = entity->pNext;
	}
		
	return dataPtr;	
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
