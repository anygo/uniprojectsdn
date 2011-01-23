#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>

#define BUFFER 1048576
#define ASCII 256


/* -----------------*
 * GLOBAL VARIABLES *
 * -----------------*/

/* Array von Pointern auf Pointer auf char-Pointer
 * wird in der main() initialisiert: */
static unsigned char ***bucketArr;

/* Array von Pointern, wo die Anzahl der Strings
 * pro Bucket gespeichert wird, wird ebenfalls
 * in der main() initialisiert: */
volatile static volatile int *pointer_indices;

/* Array, in dem die aktuelle Groesse des dazugehoerigen
 * buckets steht, Initialisierung in der main(): */
static int *pointer_indices_MAX;

/* Die Groesse des eingelesenen Files, wird
 * in readInput() initialisiert: */
static long filesize;

/* */
volatile static int bucketsTouched = 32;

/* Array, dass am Ende mit write() ausgegeben
 * werden soll, wird in readInput() initialisiert: */
static unsigned char *output;


/* ---- *
 * SORT *
 * ---- */

extern int mySSE4strcmp (unsigned const char * s1, unsigned const char * s2);

int my_strcmp (unsigned const char * s1,unsigned const char * s2) 
{
   while (*s1 == *s2) {
      s1++;
      s2++;
	  if (*s1 == 0)
	  	return -1;
	  if (*s2 == 0)
	  	return 1;
   }   
   return *s1 - *s2;
}


static int mystrcmp(const void *p1, const void *p2)
{	
	return strcmp(* (unsigned char * const *) p1, * (unsigned char * const *) p2);
}

void * sortThread1()
{
	int bT = 32;
	while (bT < ASCII)
	{
		qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char *), mystrcmp);
		bT += 4;
	}
}
void * sortThread2()
{
	int bT = 33;
	while (bT < ASCII)
	{
		qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char *), mystrcmp);
		bT += 4;
	}
}
void * sortThread3()
{
	int bT = 34;
	while (bT < ASCII)
	{
		qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char *), mystrcmp);
		bT += 4;
	}
}
void * sortThread4()
{
	int bT = 35;
	while (bT < ASCII)
	{
		qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char *), mystrcmp);
		bT += 4;
	}
}

void sort()
{
	pthread_t t1;
	pthread_t t2;
	pthread_t t3;
	pthread_t t4;

	pthread_create(&t1, NULL, sortThread1, NULL);
	pthread_create(&t2, NULL, sortThread2, NULL);
	pthread_create(&t3, NULL, sortThread3, NULL);
	pthread_create(&t4, NULL, sortThread4, NULL);

	pthread_join(t1, NULL);
	pthread_join(t2, NULL);
	pthread_join(t3, NULL);
	pthread_join(t4, NULL);
}


/* ---------- *
 * READ INPUT *
 * ---------- */

void addToBucket(unsigned char *str)
{
	int tmp = (int)str[0];
	bucketArr[tmp][pointer_indices[tmp]++] = str;
}

void readInput()
{
	struct stat info;
	unsigned char *input;
	unsigned char cur;
	unsigned char next;
	int ccount = 0;
	int curstrlen = 0;  /* Laenge des Strings, der in ein Bucket gespeichert wird 
						 * noch nicht eingebaut */

	fstat(STDIN_FILENO, &info);
	filesize = info.st_size;

	if (filesize < 1) 
		exit(EXIT_SUCCESS);

	input = (unsigned char*)malloc(sizeof(unsigned char)*filesize);
	output = (unsigned char*)malloc(sizeof(unsigned char)*filesize);
	input = mmap(0, filesize, PROT_WRITE, MAP_PRIVATE, STDIN_FILENO, 0);
	
	
	while (ccount < filesize && input[ccount] <= 32)
	{
		if (input[ccount] != '\0')
			return;
		/* ignoriere leere Strings und
		 * sonstigen Bloedsinn am Anfang
		 */
		ccount++;
	}

	addToBucket(&input[ccount++]);

	//while ((cur = input[ccount]) != 0)
	while (ccount <= filesize)
	{
	cur = input[ccount];
		if (cur == '\n')
		{
			input[ccount] = '\0';  /*ENTFERNE MICHHHH*/
			if ((next = input[++ccount]) >= 32)
			{
				addToBucket(&input[ccount]);
			}
			else
			{
			//	input[ccount] = '\0';
				/* nix tun, falls leere Zeile gelesen */
			}
		}
		else
		{
			ccount++;
		}
	}
}


/* ----- *
 * PRINT *
 * ----- */

void shitprint()
{
	int i, j;
	for (i = 32; i < ASCII; i++)
	{
		for (j = 0; j < pointer_indices[i]; j++)
		{
			puts(bucketArr[i][j]);
		}
	}
}

int main()
{	
	int i, j;

	/* Initialisierung der globalen Variablen */
	bucketArr = malloc(sizeof(unsigned char *)*ASCII);
	for (i = 33; i < ASCII; i++)
	{
		bucketArr[i] = malloc(sizeof(unsigned char *)*BUFFER);
	}
	pointer_indices = (int *)malloc(sizeof(int *)*ASCII);
	pointer_indices_MAX = (int *)malloc(sizeof(int *)*ASCII);
	/* -------------------------------------- */
	
	
	/* Auf gehts */
	readInput();
	sort();
	shitprint();
	return(EXIT_SUCCESS);
}
