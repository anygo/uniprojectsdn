#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>


#define SMALL_BUFFER  1048576
#define ZEICHEN 256

unsigned char* bucketArr[ZEICHEN][SMALL_BUFFER];
long long pointer_indices[ZEICHEN];
static long size;
static int bucketsTouched = 33;

static int mystrcmp(const void *p1, const void *p2)
{
    return my_strcmp(* (unsigned char * const *) p1, * (unsigned char * const *) p2);
}

/* eigene Strcmp version */
int my_strcmp (unsigned const char * s1,unsigned const char * s2)
{
   while (*s1 == *s2) {
      s1++;
      s2++;
	  if(*s1 == '\n' || *s2 == '\n') return 0;
   }
   return *s1 - *s2;
}

void addToBucket(unsigned char* string)
{
	int tmp = (int)string[0];
	bucketArr[tmp][pointer_indices[tmp]++] = string;
}

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
	
	/* pthread await signal */
	/*  copy sorted string in one output array  */
	
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
	int i = 0; int j; int k = 0;

	int l = 0;
	unsigned char* output = (unsigned char*)malloc(sizeof(unsigned char)*(size+10));
	
	for (i = 33; i < ZEICHEN; i++)
	{	
		if(k >= size) break;
		for (j = 0; j < pointer_indices[i]; j++)
		{	
		
			//fputs(bucketArr[i][j],stdout);
				l = 0;
				while(bucketArr[i][j][l] != '\n')
					output[k++] = bucketArr[i][j][l++];
					
				if(bucketArr[i][j][l] = '\n') {
					output[k++] = bucketArr[i][j][l++];
				l++;
				}
				if(k >= size) break;
			
		}
	}  

	write(STDOUT_FILENO,output,k);
}



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
	inputArr = mmap(0, info.st_size, PROT_WRITE, MAP_PRIVATE, STDIN_FILENO, 0);
	
	
	addToBucket(&inputArr[ccount++]);

	while((c = inputArr[ccount]) != '\0'){	
		if(c == '\n') 
		{
			//inputArr[ccount] = '\0';
			addToBucket(&inputArr[++ccount]);
		} 
		else 
		{
			ccount++;
		}
	}
	/* Nicht unbedingt noetig but u never know...hatte da uachmal ein \n drinstehen */
	inputArr[ccount] = '\0';
}
	

int main()
{
    readInput();
	//perror("read compl");
    sort();
	//perror("sort done");
    print();
	//perror("complete");
    return 0;
}
