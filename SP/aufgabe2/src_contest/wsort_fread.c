#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define STRBUFFER 1048576

static long size;
static int strCount = 0;

char** readInput()
{
	struct stat info;
	char* inputArr;
	int charCount = 0;
	int t = 1;
	char** arr;
	char tmp;

	fstat(STDIN_FILENO, &info);
	inputArr = (char*)malloc(sizeof(char)*info.st_size);
	
	size = info.st_size;
	
	fread((void*)inputArr, sizeof(char), info.st_size-1, stdin);

	arr = (char**)malloc(sizeof(char*)*STRBUFFER);
	if (inputArr[0] == '\n')
	{
		inputArr[0] = '\n';
		arr[strCount] = &inputArr[charCount];
		strCount++;
		charCount++;
	}
	arr[strCount] = &inputArr[charCount];
	strCount++;
	charCount++;
	
	while ((tmp = inputArr[charCount]) != '\0')
	{
		if (tmp == '\n')
		{
			/*inputArr[charCount] = '\0';*/
			arr[strCount] = &inputArr[++charCount];
			strCount++;
			if (strCount > t*STRBUFFER)
			{
				t++;
				arr = realloc(arr, sizeof(char*)*STRBUFFER*t);
			}
		}
		else
		{
			charCount++;
		}
	}
	strCount;
	return arr;
}

static int mystrcmp(const void *p1, const void *p2)
{
	return strcmp(* (char * const *) p1, * (char * const *) p2);
}

void printArr(char** arr)
{	
	write(STDOUT_FILENO, arr[0], size);
}

int main()
{	
	char** arr;
	
	arr = readInput();
	
	/*qsort(arr, strCount, sizeof(char*), mystrcmp);*/
	printArr(arr);
	return 0;
}
