#include "xmlparser.h"

char *InnerText(char *inner, char *pText, char *beginMark, char *endMark) 
{
	char *beginStart = strstr(pText, beginMark);
	if (beginStart == NULL) 
		return NULL;
	char *beginEnd = beginStart + strlen(beginMark);
	char *endStart = strstr(beginEnd, endMark);
	if (endStart < 0) 
		return NULL;
	int len = endStart-beginEnd;
	strncpy(inner, beginEnd, len);
	inner[len] = '\0';
	return inner;
}

int FileToStr(char *buffer, char *fileName, int *sizePtr) 
{
	printf("fileName = %s\n", fileName);
	FILE *fp;
	if((fp = fopen(fileName, "rb")) == NULL){
		printf("File not found!\n");
		assert(fp);
		return -1;
	}
	fseek(fp, 0 , SEEK_END);
	long size = ftell(fp);
	rewind(fp);
	fread(buffer,size,1,fp);
	fclose(fp);
	return 0;
}
