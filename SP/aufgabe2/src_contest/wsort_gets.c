#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STRBUFFER 65536

static int strCount = 0;

char** readInput()
{
	char** inputArr;
	char string[101];
	int strBufferCount = 1;

	inputArr = (char**)malloc(sizeof(char*)*STRBUFFER);
	
	while (gets(string))
	{
		int len = strlen(string);
		if (strCount >= strBufferCount*STRBUFFER)
			inputArr = realloc(inputArr, sizeof(char*)*((++strBufferCount)*STRBUFFER));
		inputArr[strCount] = (char*)malloc(sizeof(char)*(len+1));
		strcpy(inputArr[strCount++], string);
	}

	return inputArr;
}

void printArr(char** arr)
{
	int i;
	for	(i = 0; i < strCount; i++)
	{
		puts(arr[i]);
	}
}

static int mystrcmp(const void *p1, const void *p2)
{
	return strcmp(* (char * const *) p1, * (char * const *) p2);
}

int main(int argc, char *argv[])
{	
	char** inputArr;

	inputArr = readInput();
	/*qsort(inputArr, strCount, sizeof(char *), mystrcmp);
	printArr(inputArr);*/
	return 0;
}
