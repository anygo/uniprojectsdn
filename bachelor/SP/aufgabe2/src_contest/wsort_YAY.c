#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>

#define BUFFER 536870912
#define SMALL_BUFFER 134217728
#define ZEICHEN 256

static int mystrcmp(const void *p1, const void *p2)
{
    return strcmp(* (char * const *) p1, * (char * const *) p2);
}

char* bucketArr[ZEICHEN][SMALL_BUFFER/128];
int pointer_indices[ZEICHEN];


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
		//qsort(bucketArr[i], pointer_indices[i], sizeof(char*), mystrcmp);
		for (j = 0; j < pointer_indices[i]; j++)
		{
			printf("%s\n", bucketArr[i][j]);
		}
	}
}


static long size;
static int strCount = 0;

char** readInput()
{
    struct stat info;
    char* inputArr;
    char** arr;
	char c;
	int ccount = 0;
	int t = 1;
    fstat(STDIN_FILENO, &info);
	size = info.st_size;

	if (size < 1) exit(0);

    inputArr = (char*)malloc(sizeof(char)*size);
	inputArr = mmap(0, info.st_size-1, PROT_WRITE, MAP_PRIVATE, STDIN_FILENO, 0);
	
	arr = (char**) malloc(sizeof(char*)*BUFFER);
	
	addToBucket(&inputArr[ccount]);
	arr[strCount++] = &inputArr[ccount++];
	
	while((c = inputArr[ccount]) != '\0'){	
		if(c == '\n') 
		{
			inputArr[ccount] = '\0';
			arr[strCount++] = &inputArr[++ccount];
			addToBucket(&inputArr[ccount]);
		
		} 
		else 
		{
			ccount++;
		}
	}
    return arr;
}

void printArr(char** arr)
{
   int j;
	for(j = 0; j < strCount; j++) {

		//fputs(arr[j],stdout);
		printf("%s\n",arr[j]);
	}
   /*write(STDOUT_FILENO, arr[0], size);*/
}

int main()
{
    char** arr;
	int i = 0;
    arr = readInput();
	sort();	
    //qsort(arr, strCount, sizeof(char*), mystrcmp);
	
//    printArr(arr);
    return 0;
}

