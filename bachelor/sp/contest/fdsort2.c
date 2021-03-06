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

static int anzahlstrings0 = 0;
static int anzahlstrings1 = 0;

static unsigned char** output1;
static unsigned char** output2;

unsigned char *input;

/* Array von Pointern auf Pointer auf char-Pointer
 * wird in der main() initialisiert: */
 
static unsigned char ***bucketArr0;
static unsigned char ***bucketArr1;
static unsigned char ***bucketArr2;
static unsigned char ***bucketArr3;


static int start0, start1, start2, start3;

static int part;
/* Array von Pointern, wo die Anzahl der Strings
 * pro Bucket gespeichert wird, wird ebenfalls
 * in der main() initialisiert: */
 
static volatile int *pointer_indices0;
static volatile int *pointer_indices1;
static volatile int *pointer_indices2;
static volatile int *pointer_indices3;


/* Die Groesse des eingelesenen Files, wird
 * in readInput() initialisiert: */
 
static long filesize;

/* */

static int bucketsTouched0 = 32;
static int bucketsTouched1 = 32;
static int bucketsTouched2 = 32;
static int bucketsTouched3 = 32;

/* Array, dass am Ende mit write() ausgegeben
 * werden soll, wird in readInput() initialisiert: */
static unsigned char *output;


/* ---- *
 * SORT *
 * ---- */

static int my_strcmp (unsigned const char * s1,unsigned const char * s2)
{
   while (*s1 == *s2) {
      s1++;
      s2++;
	  if(*s1 == '\n') return 0;
   }
   return *s1 - *s2;
}

static int my_strcmp_2 (unsigned const char * s1,unsigned const char * s2)
{
	int ret;
	int length = 0;
	while (*s1 == *s2) {
		length++;
	//	printf("%c;%c\n", *s1, *s2);
		s1++;
		s2++;
		if(*s1 == '\n') return -(++length);
	}
	ret = *s1 - *s2;
	if (ret < 0)
	{
		while (*s1 != '\n')
		{
			s1++;
			length++;
		}
		length++;
		return -length;
	}
	else
	{
		while (*s2 != '\n')
		{
			s2++;
			length++;
		}
		length++;
		return length;
	}
   
}

static int mystrcmp(const void *p1, const void *p2)
{
	return my_strcmp(* (unsigned char * const *) p1+1, * (unsigned char * const *) p2+1);
}

/* ----- *
 * MERGE *
 * ----- */

void merge0()
{
	int i, j = 0, k = 0, l = 0;
	for(i=33; i < ASCII; i++)
	{
		anzahlstrings0 = anzahlstrings0 + pointer_indices0[i] + pointer_indices1[i];
	}
	output1 = (unsigned char **)malloc(sizeof(unsigned char *)*anzahlstrings0);

	for (i = 33; i < ASCII; i++)
	{
		j = 0;
		k = 0;
		if (pointer_indices0[i] <= pointer_indices1[i])
		{
		
			while (j < pointer_indices0[i])
			{
			if(bucketArr1[i][k] == NULL) break;
			//printf("i: %d j: %d k: %d b0 %c b1 %c bla\n",i,j,k,bucketArr0[i][j][0],bucketArr1[i][j][0]);
				if (my_strcmp(bucketArr0[i][j], bucketArr1[i][k]) < 0)
				{
					output1[l++] = bucketArr0[i][j++];
					
				}
				else
				{
					output1[l++] = bucketArr1[i][k++];
				}
			}
			while (k < pointer_indices1[i])
			{
				output1[l++] = bucketArr1[i][k++];
			}
		}
		else
		{
			
			while (k < pointer_indices1[i])
			{
			//printf("i: %d j: %d k: %d b0 %c b1 %c bla\n",i,j,k,bucketArr0[i][j][0],bucketArr1[i][j][0]);
			if(bucketArr0[i][j] == NULL) break;
				if (my_strcmp(bucketArr1[i][k], bucketArr0[i][j]) < 0)
				{
					output1[l++] = bucketArr1[i][k++];
					
				}
				else
				{
					output1[l++] = bucketArr0[i][j++];
				}
			}
			while (j < pointer_indices0[i])
			{
				output1[l++] = bucketArr0[i][j++];
			}
		}
	}	
//	printf("Merge 0 Strings %d\n",l);
}

void merge1()
{
	int i, j = 0;
	int k = 0;
	int l = 0;
	for(i=33; i < ASCII; i++)
	{
		anzahlstrings1 = anzahlstrings1 + pointer_indices2[i] + pointer_indices3[i];
	}
	output2 = (unsigned char **)malloc(sizeof(unsigned char*)*anzahlstrings1+sizeof(unsigned char*));

	for (i = 33; i < ASCII; i++)
	{
		j = 0;
		k = 0;
		
		if (pointer_indices2[i] <= pointer_indices3[i])
		{
		
			while (j < pointer_indices2[i])
			{
			if(bucketArr3[i][k] == NULL) break;
		//	printf("%d %d \n",pointer_indices2[i],pointer_indices3[i]);
		//	printf("i: %d j: %d k: %d b0 %c b1 %c bla\n",i,j,k,bucketArr2[i][j][0],bucketArr3[i][k][0]);
				if (my_strcmp(bucketArr2[i][j], bucketArr3[i][k]) < 0)
				{
					output2[l++] = bucketArr2[i][j++];
				}
				else
				{
					output2[l++] = bucketArr3[i][k++];
				}
			}
			while (k < pointer_indices3[i])
			{
				output2[l++] = bucketArr3[i][k++];
			}
		}
		else
		{
		
			while (k < pointer_indices3[i])
			{
				if(bucketArr2[i][j] == NULL) break;
				if (my_strcmp(bucketArr3[i][k], bucketArr2[i][j]) < 0)
				{
					output2[l++] = bucketArr3[i][k++];
				}
				else
				{
					output2[l++] = bucketArr2[i][j++];
				}
			}
			while (j < pointer_indices2[i])
			{
				output2[l++] = bucketArr2[i][j++];
			}
		}
	}
/*	printf("%s\n", output2[0] );
	printf("%s\n", output1[0] );
	printf("Merge 1 Strings %d\n",l);
	printf("Str 1: %d Str 2: %d\n",anzahlstrings0,anzahlstrings1);
	*/
}

int my_strlen(char * str)
{
	int length = 0;
	while (*str != '\n')
		length++;
	return ++length;
}

void * printmerge()
{
	int i = 0, j = 0, k = 0;
	int length;
	
	if (anzahlstrings0 < anzahlstrings1)
	{
		while (i < anzahlstrings0)
		{
			if(output2[j] == NULL) break;
			if ((length = my_strcmp_2(output1[i], output2[j])) < 0)
			{
				strncpy(&output[k], output1[i++], -length);
				k = k - length;
			}
			else
			{
				strncpy(&output[k], output2[j++], length);
				k = k + length;
			}
		
		}
		while (j < anzahlstrings1)
		{
			strncpy(&output[k], output2[j], (length = my_strlen(output2[j])));
			j++;
			k = k + length;
		}

	}
	else
	{
		while (j < anzahlstrings1)
		{
			/*if(output1[i] == NULL || output2[j] == NULL) {
			//printf("i: %d j: %d",i,j);
			perror("Hilfe");
			//exit(1);
			}
			*/
			if(output1[i] == NULL) break;
			if ((length = my_strcmp_2(output1[i], output2[j])) < 0)
			{
				strncpy(&output[k], output1[i++], -length);
				k = k - length;
				if(length > 100) perror("test");
			}
			else
			{
				strncpy(&output[k], output2[j++], length);
				k = k + length;
			}
		}
		while (i < anzahlstrings0)
		{
			strncpy(&output[k], output1[i], (length = my_strlen(output1[i])));
			i++;
			k = k + length;
		}
	}
	write(STDOUT_FILENO, output, k);
}


/* ---------- *
 * READ INPUT *
 * ---------- */

void sort0() 
{
	int bT;
	while ((bT = bucketsTouched0++) < ASCII)
	{
		qsort(bucketArr0[bT], pointer_indices0[bT], sizeof(unsigned char *), mystrcmp);
	}
}

void sort1() 
{
	int bT;
	while ((bT = bucketsTouched1++) < ASCII)
	{
		qsort(bucketArr1[bT], pointer_indices1[bT], sizeof(unsigned char *), mystrcmp);
	}
}

void sort2() 
{
	int bT;
	while ((bT = bucketsTouched2++) < ASCII)
	{
		qsort(bucketArr2[bT], pointer_indices2[bT], sizeof(unsigned char *), mystrcmp);
	}
}

void sort3() 
{
	int bT;
	while ((bT = bucketsTouched3++) < ASCII)
	{
		qsort(bucketArr3[bT], pointer_indices3[bT], sizeof(unsigned char *), mystrcmp);
	}
}

void * thread0(void *bla) 
{	
	int ccount = start0;
	int tmp;

	tmp = (int)input[ccount];
	bucketArr0[tmp][pointer_indices0[tmp]++] = &input[ccount++];
	while (ccount < start1)
	{
		if (input[ccount] == '\n')
		{
			//input[ccount] = '\0';
			if (input[++ccount] >= 32)
			{
				tmp = (int)input[ccount];
				bucketArr0[tmp][pointer_indices0[tmp]++] = &input[ccount++];
			}
		}
		else
		{
			ccount++;
		}
	}

	sort0();
}

void * thread1(void *bla) 
{	
	int ccount = start1;
	int tmp;

	tmp = (int)input[ccount];
	bucketArr1[tmp][pointer_indices1[tmp]++] = &input[ccount++];
	while (ccount < start2)
	{
		if (input[ccount] == '\n')
		{
			//input[ccount] = '\0';
			if (input[++ccount] >= 32)
			{
				tmp = (int)input[ccount];
				bucketArr1[tmp][pointer_indices1[tmp]++] = &input[ccount++];
			}
		}
		else
		{
			ccount++;
		}
	}

	sort1();
}

void * thread2(void *bla) 
{	
	int ccount = start2;
	int tmp;

	tmp = (int)input[ccount];
	bucketArr2[tmp][pointer_indices2[tmp]++] = &input[ccount++];
	while (ccount < start3)
	{
		if (input[ccount] == '\n')
		{
			if (input[++ccount] >= 32)
			{
				tmp = (int)input[ccount];
				bucketArr2[tmp][pointer_indices2[tmp]++] = &input[ccount++];
			}
		}
		else
		{
			ccount++;
		}
	}

	sort2();
}

void * thread3(void *bla) 
{	
	int ccount = start3;
	int tmp;

	tmp = (int)input[ccount];
	bucketArr3[tmp][pointer_indices3[tmp]++] = &input[ccount++];
	while (ccount <= filesize)
	{
		if (input[ccount] == '\n')
		{
			if (input[++ccount] >= 32)
			{
				tmp = (int)input[ccount];
				bucketArr3[tmp][pointer_indices3[tmp]++] = &input[ccount++];
			}
		}
		else
		{
			ccount++;
		}
	}

	sort3();
}



void readInput()
{
	struct stat info;
	int ccount = 0;
	int part;

	fstat(STDIN_FILENO, &info);
	filesize = info.st_size;

	if (filesize < 1) 
		exit(EXIT_SUCCESS);

	input = (unsigned char*)malloc(sizeof(unsigned char)*filesize);
	output = (unsigned char*)malloc(sizeof(unsigned char)*filesize);
	input = mmap(0, filesize, PROT_WRITE, MAP_PRIVATE, STDIN_FILENO, 0);
	
	part = filesize/4;

	while (ccount < filesize, input[ccount] <= 32)
	{
		/* ignoriere leere Strings und
		 * sonstigen Bloedsinn am Anfang
		 */
		ccount++;
	}
	//input[ccount] = '\0';
	start0 = ccount;
	ccount = part;
	while (input[ccount++] != '\n')
	{
		//nix tun
	}
	//input[ccount-1] = '\0';
	start1 = ccount + 1;
	ccount = part*2;
	while (input[ccount++] != '\n')
	{
		//nix tun
	}
	//input[ccount-1] = '\0';
	start2 = ccount + 1;
	ccount = part*3;
	while (input[ccount++] != '\n')
	{
		//nix tun
	}
	//input[ccount-1] = '\0';
	start3 = ccount + 1;

	pthread_t t1;
	pthread_t t2;
	pthread_t t3;
	pthread_t t4;

	pthread_create(&t1, NULL, thread0, NULL);
	pthread_create(&t2, NULL, thread1, NULL);
	pthread_create(&t3, NULL, thread2, NULL);
	pthread_create(&t4, NULL, thread3, NULL);
	
	pthread_join(t1, NULL);
	pthread_join(t2, NULL);
	pthread_join(t3, NULL);
	pthread_join(t4, NULL);	
}



/* ----- *
 * PRINT *
 * ----- 

		void shitprint()
		{
			int i, j,l;
			int k = 0;
			for (i = 32; i < ASCII; i++)
			{

				for (j = 0; j < pointer_indices[i]; j++)
				{
						l = 0;
						while(bucketArr[i][j][l] != '\n')
							output[k++] = bucketArr[i][j][l++];
							
						if(bucketArr[i][j][l] = '\n') {
							output[k++] = bucketArr[i][j][l++];
						l++;
						}
						//if(k >= filesize) break;
			
				}
			}
			write(STDOUT_FILENO,output,k);
		}
*/
int main()
{	
	int i;

	/* Initialisierung der globalen Variablen */
	bucketArr0 = (unsigned char ***)malloc(sizeof(unsigned char *)*ASCII);
	for (i = 33; i < ASCII; i++)
	{
		bucketArr0[i] = (unsigned char **)malloc(sizeof(unsigned char *)*BUFFER);
	}
		bucketArr1 = (unsigned char ***)malloc(sizeof(unsigned char *)*ASCII);
	for (i = 33; i < ASCII; i++)
	{
		bucketArr1[i] = (unsigned char **)malloc(sizeof(unsigned char *)*BUFFER);
	}
		bucketArr2 = (unsigned char ***)malloc(sizeof(unsigned char *)*ASCII);
	for (i = 33; i < ASCII; i++)
	{
		bucketArr2[i] = (unsigned char **)malloc(sizeof(unsigned char *)*BUFFER);
	}
		bucketArr3 = (unsigned char ***)malloc(sizeof(unsigned char *)*ASCII);
	for (i = 33; i < ASCII; i++)
	{
		bucketArr3[i] = (unsigned char **)malloc(sizeof(unsigned char *)*BUFFER);
	}
	pointer_indices0 = (int *)malloc(sizeof(int)*ASCII);
	pointer_indices1 = (int *)malloc(sizeof(int)*ASCII);
	pointer_indices2 = (int *)malloc(sizeof(int)*ASCII);
	pointer_indices3 = (int *)malloc(sizeof(int)*ASCII);
	
	/* -------------------------------------- */
	
	unsigned char arr[] = {'d','u','m','b','\n'}; 
	     unsigned char arr1[] = {'d','u','m','b','e','r','\n'};
	printf("CMP: %d\n",my_strcmp_2(arr1,arr)); 
	      printf("CMP: %d\n",my_strcmp_2(arr,arr1));



	/* Auf gehts */
	readInput();
	
	merge0();
	merge1();
	
	printmerge();

	return(EXIT_SUCCESS);
}
