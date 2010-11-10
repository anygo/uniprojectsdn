#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>


#define SMALL_BUFFER 1048576
#define ZEICHEN 256

static int mystrcmp(const void *p1, const void *p2)
{
    return strcmp(* (unsigned char * const *) p1+1, * (unsigned char * const *) p2+1);
}

unsigned char* bucketArr[ZEICHEN][SMALL_BUFFER];
long long pointer_indices[ZEICHEN];


void addToBucket(unsigned char* string)
{
	int tmp = (int)string[0];
	bucketArr[tmp][pointer_indices[tmp]++] = string;
}


static int bucketsTouched = 33;

/* ------------------------------------------------------------------------ */
void* sortThread(void* arg)
{
	int bT;
	while ((bT = bucketsTouched++) < ZEICHEN-1)
	{
		qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char*), mystrcmp);
	}
	return NULL;
}

void sort()
{
	pthread_t t1;
	pthread_t t2;
	pthread_t t3;
	pthread_t t4;
//	pthread_t t5;
//	pthread_t t6;
//	pthread_t t7;
//	pthread_t t8;

	
	pthread_create(&t1, NULL, sortThread, NULL);
	pthread_create(&t2, NULL, sortThread, NULL);
	pthread_create(&t3, NULL, sortThread, NULL);
	pthread_create(&t4, NULL, sortThread, NULL);
//	pthread_create(&t5, NULL, sortThread, NULL);
//	pthread_create(&t6, NULL, sortThread, NULL);
//	pthread_create(&t7, NULL, sortThread, NULL);
//	pthread_create(&t8, NULL, sortThread, NULL);
	
	pthread_join(t1, NULL);
	pthread_join(t2, NULL);
	pthread_join(t3, NULL);
	pthread_join(t4, NULL);
//	pthread_join(t5, NULL);
//	pthread_join(t6, NULL);
//	pthread_join(t7, NULL);
//	pthread_join(t8, NULL);	
}
/* ------------------------------------------------------------------------ */

static void print()
{
	int i, j;
	for (i = 33; i < ZEICHEN; i++)
	{	
		for (j = 0; j < pointer_indices[i]; j++)
		{
			puts(bucketArr[i][j]);
		}
	}
}

static long size;

void readInput()
{
    struct stat info;
    unsigned char* inputArr;
	char c;
	int ccount = 0;
    fstat(STDIN_FILENO, &info);
	size = info.st_size;

	if (size < 1) exit(0);

    inputArr = (unsigned char*)malloc(sizeof(unsigned char)*size);
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
}

int main()
{
    readInput();
	sort();
	print();
    return 0;
}

