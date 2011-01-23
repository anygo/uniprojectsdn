#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>


#define BUFFER 536870912
#define SMALL_BUFFER 134217728
#define ZEICHEN 256

static int mystrcmp(const void *p1, const void *p2)
{
    return strcmp(* (char * const *) p1, * (char * const *) p2);
}

char* bucketArr[ZEICHEN][SMALL_BUFFER/96];
long long pointer_indices[ZEICHEN];


void addToBucket(char* string)
{
	int tmp = (int)string[0]+95;
	bucketArr[tmp][pointer_indices[tmp]++] = string;
}

void sort()
{
	int i;
	int j;
	for (i = 0; i < 256; i++)
	{	
		qsort(bucketArr[i], pointer_indices[i], sizeof(char*), mystrcmp);
		for (j = 0; j < pointer_indices[i]; j++)
		{
			//printf("%s\n", bucketArr[i][j]);
			puts(bucketArr[i][j]);
		}
	}
}

static int bucketsTouched = 0;

/* ------------------------------------------------------------------------ */
void* sortThread(void* arg)
{
	int bT = bucketsTouched++;
	qsort(bucketArr[bT], pointer_indices[bT], sizeof(char*), mystrcmp);
}
/* ------------------------------------------------------------------------ */

static long size;

char** readInput()
{
    struct stat info;
    char* inputArr;
    char** arr;
	char c;
	int ccount = 0;
    fstat(STDIN_FILENO, &info);
	size = info.st_size;

	if (size < 1) exit(0);

    inputArr = (char*)malloc(sizeof(char)*size);
	inputArr = mmap(0, info.st_size-1, PROT_WRITE, MAP_PRIVATE, STDIN_FILENO, 0);
	
	
	addToBucket(&inputArr[ccount++]);

	while((c = inputArr[ccount]) != '\0'){	
		if(c == '\n') 
		{
			inputArr[ccount] = '\0';
			addToBucket(&inputArr[++ccount]);
		} 
		else 
		{
			ccount++;
		}
	}
    return arr;
}

int main()
{
    char** arr;
    arr = readInput();
	sort();	
    return 0;
}

