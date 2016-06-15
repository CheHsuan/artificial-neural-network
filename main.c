#include <stdio.h>
#include "src/neuralnet.h"

int main(int argc, char *argv[])
{
	extern NET_DEFINE netDefinition;
	
	LoadNetDefinition(argv[1]);
	printf("netDefinition:\n");
	printf("\tLearning Rate: %f\n", netDefinition.learningRate);
	printf("\tEpoch: %d\n", netDefinition.epoch);
	printf("\tInput Layer Neuron Number: %d\n", netDefinition.inputLayerNeuronNum);
	printf("\tHidden Layer Neuron Number: %d\n", netDefinition.hiddenLayerNeuronNum);
	printf("\tOutput Layer Neuron Number: %d\n", netDefinition.outputLayerNeuronNum);
	printf("\tActivation Function: %s\n", netDefinition.activationFunction);
	printf("\tWeight Assignment Method: %s\n", netDefinition.weightAssignment);
	LoadTrainingSet(argv[2]);
	LoadValidationSet(argv[3]);
	Training();
	FreeMemory();
	
	return 0;
}
