#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>

#define STRBUFFER 536870912

static long size;
static int strCount = 0;

char** readInput()
{
	struct stat info;
	char* inputArr;
	int charCount = 0;
	char** arr;
	char tmp;

	fstat(STDIN_FILENO, &info);
	size = info.st_size;

	inputArr = (char*)malloc(sizeof(char)*info.st_size);

	inputArr = mmap(0, info.st_size, PROT_WRITE, MAP_PRIVATE, STDIN_FILENO, 0);

	arr = (char**)malloc(sizeof(char*)*STRBUFFER);
	
	while (inputArr[charCount] != '\0')
	{
		if (tmp = '\n')
		{
			inputArr[charCount++] = '\0';
			arr[strCount++] = &inputArr[charCount];
			// in Bucket werfen
		}
		else
		{
			charCount++;
		}
	}

	return arr;
}

static int mystrcmp(const void *p1, const void *p2)
{
	return strcmp(* (char * const *) p1, * (char * const *) p2);
}

void printArr(char** arr)
{
	int i = 0;
	int j = 0;
	char c;
	for (i = 1; i < strCount; i++)
	{
		while ((c = arr[i][j++]) != '\0')
		{
			putchar(c);
		}
		putchar(c);
		j=0;
	}

//	write(STDOUT_FILENO, arr[1], size);
}

int main()
{	
	char** arr;
	
	arr = readInput();
	//qsort(arr, strCount, sizeof(char*), mystrcmp);
	//printArr(arr);
	return 0;
}
