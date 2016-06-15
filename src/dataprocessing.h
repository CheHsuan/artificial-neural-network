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
void Free2DMemory(double **, int);
void FreeDataList(ENTITY *);
void FreeMemory();
