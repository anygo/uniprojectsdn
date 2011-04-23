#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/mman.h>
#include <pthread.h>

#define BUFFER 1048576
#define ASCII 256

#define swap2(a, b) { t = *(a); *(a) = *(b); *(b) = t; }
#define ptr2char(i) (*(*(i) + depth))
#define med3(a, b, c) med3func(a, b, c, depth)

/* -----------------*
 * GLOBAL VARIABLES *
 * -----------------*/

/* Array von Pointern auf Pointer auf char-Pointer
 * wird in der main() initialisiert: */
static unsigned char ***bucketArr;

/* Array von Pointern, wo die Anzahl der Strings
 * pro Bucket gespeichert wird, wird ebenfalls
 * in der main() initialisiert: */
static volatile int *pointer_indices;

/* Array, in dem die aktuelle Groesse des dazugehoerigen
 * buckets steht, Initialisierung in der main(): */
static int *pointer_indices_MAX;

/* Die Groesse des eingelesenen Files, wird
 * in readInput() initialisiert: */
static long filesize;

/* */
static int bucketsTouched = 32;

/* Array, dass am Ende mit write() ausgegeben
 * werden soll, wird in readInput() initialisiert: */
static unsigned char *output;


/* ---- *
 * SORT *
 * ---- */
 
 
 void vecswap2(unsigned char **a, unsigned char **b, int n)
{   while (n-- > 0) {
        unsigned char *t = *a;
        *a++ = *b;
        *b++ = t;
    }
}

int min(int a, int b)
{
  return a<b ? a : b;
}


unsigned char **med3func(unsigned char **a, unsigned char **b, unsigned char **c, int depth)
{   int va, vb, vc;
    if ((va=ptr2char(a)) == (vb=ptr2char(b)))
        return a;
    if ((vc=ptr2char(c)) == va || vc == vb)
        return c;       
    return va < vb ?
          (vb < vc ? b : (va < vc ? c : a ) )
        : (vb > vc ? b : (va < vc ? a : c ) );
}


void inssort(unsigned char **a, int n, int d)
{   unsigned char **pi, **pj, *s, *t;
    for (pi = a + 1; --n > 0; pi++)
        for (pj = pi; pj > a; pj--) {
            for (s=*(pj-1)+d, t=*pj+d; *s==*t && *s!=0; s++, t++)
                ;
            if (*s <= *t)
                break;
            swap2(pj, pj-1);
    }
}

void ssort2(unsigned char **a, int n, int depth)
{   int d, r, partval;
    unsigned char **pa, **pb, **pc, **pd, **pl, **pm, **pn, *t;
    if (n < 10) {
        inssort(a, n, depth);
        return;
    }
    pl = a;
    pm = a + (n/2);
    pn = a + (n-1);
    if (n > 30) { 
        d = (n/8);
        pl = med3(pl, pl+d, pl+2*d);
        pm = med3(pm-d, pm, pm+d);
        pn = med3(pn-2*d, pn-d, pn);
    }
    pm = med3(pl, pm, pn);
    swap2(a, pm);
    partval = ptr2char(a);
    pa = pb = a + 1;
    pc = pd = a + n-1;
    for (;;) {
        while (pb <= pc && (r = ptr2char(pb)-partval) <= 0) {
            if (r == 0) { swap2(pa, pb); pa++; }
            pb++;
        }
        while (pb <= pc && (r = ptr2char(pc)-partval) >= 0) {
            if (r == 0) { swap2(pc, pd); pd--; }
            pc--;
        }
        if (pb > pc) break;
        swap2(pb, pc);
        pb++;
        pc--;
    }
    pn = a + n;
    r = min(pa-a, pb-pa);    vecswap2(a,  pb-r, r);
    r = min(pd-pc, pn-pd-1); vecswap2(pb, pn-r, r);
    if ((r = pb-pa) > 1)
        ssort2(a, r, depth);
    if (ptr2char(a + r) != 0)
        ssort2(a + r, pa-a + pn-pd-1, depth+1);
    if ((r = pd-pc) > 1)
        ssort2(a + n-r, r, depth);
}

void ssort2main(unsigned char **a, int n) { ssort2(a, n, 0); }
 

int my_strcmp (unsigned const char * s1,unsigned const char * s2) 
{
   while (*s1 == *s2) {
      s1++;
      s2++;
      if(*s1 == '\0' || *s2 == '\0') 
	  	return 0;
   }   
   return *s1 - *s2;
}


static int mystrcmp(const void *p1, const void *p2)
{
	return strcmp(* (unsigned char * const *) p1, * (unsigned char * const *) p2);
}

void * sortThread()
{
	int bT;
	while ((bT = bucketsTouched++) < ASCII)
	{
		qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char *), mystrcmp);
	}
}


void * sortThread1()
{
      int bT = 32;
      while (bT  < ASCII)
      {
		ssort2main(bucketArr[bT], pointer_indices[bT]);
          //qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char *), mystrcmp);
          bT += 4;
      }
}

void * sortThread2()
{
      int bT = 33;
      while (bT  < ASCII)
      {
			ssort2main(bucketArr[bT], pointer_indices[bT]);
          //qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char *), mystrcmp);
          bT += 4;
      }
}

void * sortThread3()
{
      int bT = 34;
      while (bT  < ASCII)
      {
		ssort2main(bucketArr[bT], pointer_indices[bT]);
         // qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char *), mystrcmp);
		bT += 4;
      }
}

  void * sortThread4()
  {
      int bT = 35;
      while (bT  < ASCII)
 
     {
		ssort2main(bucketArr[bT], pointer_indices[bT]);
         // qsort(bucketArr[bT], pointer_indices[bT], sizeof(unsigned char *), mystrcmp);
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
	//if (pointer_indices[tmp] >= (pointer_indices_MAX[tmp]))
	//{
	//	bucketArr[tmp] = realloc(bucketArr[tmp], pointer_indices_MAX[tmp]+BUFFER);
	//}
	bucketArr[tmp][pointer_indices[tmp]++] = str;
}

void readInput()
{
	struct stat info;
	unsigned char *input;
	unsigned char cur;
	unsigned char next;
	int ccount = 0;

	fstat(STDIN_FILENO, &info);
	filesize = info.st_size;

	if (filesize < 1) 
		exit(EXIT_SUCCESS);

	input = (unsigned char*)malloc(sizeof(unsigned char)*filesize);
	output = (unsigned char*)malloc(sizeof(unsigned char)*filesize);
	input = mmap(0, filesize, PROT_WRITE, MAP_PRIVATE, STDIN_FILENO, 0);
	
	
	while (ccount < filesize, input[ccount] <= 32)
	{
		/* ignoriere leere Strings und
		 * sonstigen Bloedsinn am Anfang
		 */
		ccount++;
	}

	addToBucket(&input[ccount++]);

	while (ccount <= filesize)
	//while (ccount < filesize)
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

