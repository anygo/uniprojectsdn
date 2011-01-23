#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>

#define STRBUFFER 1048576

static long size;
static int strCount = 0;

char *mystrcpy(char *target, const char *source)
{
	char *orig_target = target;
	while(*source != '\n')
		*target++ = *source++;
	*target++ = '\n';
	return orig_target;
}

char** readInput()
{
	struct stat info;
	char* inputArr;
	int charCount = 0;
	int t = 1;
	char** arr;
	char tmp;
	long count = 0;
	long tSTRBUFFER = STRBUFFER;

	fstat(STDIN_FILENO, &info);
	inputArr = (char*)malloc(sizeof(char)*info.st_size);
	
	size = info.st_size;

	inputArr = mmap(0, info.st_size, PROT_READ, MAP_SHARED, STDIN_FILENO, 0);

	arr = (char**)malloc(sizeof(char*)*STRBUFFER);
	
	arr[strCount] = &inputArr[charCount];
	strCount++;
	charCount++;
	
	while (count < size)
	{
		tmp = inputArr[charCount];
		if (tmp == '\n')
		{
			arr[strCount] = &inputArr[++charCount];
			strCount++;
			if (strCount > tSTRBUFFER)
			{
				t++;
				tSTRBUFFER += STRBUFFER;
				arr = realloc(arr, sizeof(char*)*tSTRBUFFER);
			}
		}
		else
		{
			charCount++;
		}
		count++;
	}
	arr = realloc(arr, sizeof(char*)*strCount);

	return arr;
}

static int mystrcmp(const void *p1, const void *p2)
{
	return strcmp(* (char * const *) p1, * (char * const *) p2);
}

void printArr(char** arr)
{	
	char* output = (char*)malloc(sizeof(char)*size);
	int i;
	char* remember = output;


	for (i = 1; i < strCount; i++)
	{
		mystrcpy(output, arr[i]);
	}

	write(STDOUT_FILENO, remember, size);
/*	int i;
	for (i = 0; i < strCount; i++)
	{
		write(STDOUT_FILENO, arr[i], 100);
	}
*/
//	write(STDOUT_FILENO, arr[0], size);
/*    int i = 0;
	int j = 0;
    for (i = 1; i < strCount; i++)
    {   
		while (arr[i][j] != '\n')
		{
       		putc(arr[i][j++], stdout);
		}
		putc(arr[i][j], stdout);
		j = 0;
  }
*/}

int main()
{	
	char** arr = readInput();

	char* string = (char*)malloc(sizeof(char)*size);

//	mystrcpy(string, arr[3]);
	puts(string);
//	puts("\nEND\n");
	
//	qsort(arr, strCount, sizeof(char*), mystrcmp);
	
	printArr(arr);
	return 0;
}
