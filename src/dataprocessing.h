#include <sys/sysinfo.h>

#define SYS_CORE get_nprocs() 

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
ENTITY *ReadDataSet(ENTITY *, const NET_DEFINE *, char *, int *);
ENTITY *Add2List(char *, const NET_DEFINE *, ENTITY *);
ENTITY ** DivideDataSet(ENTITY *, int);
void FreeDataList(ENTITY *);
void FreeDataMemory();
