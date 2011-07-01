#ifndef DEFSRBC_H
#define DEFSRBC_H

#include <float.h>
#include <cutil_inline.h>
// C99 but not part of MS VS Compiler, again shame on MS...
// -> so were using "unsigned __int32"
//#include <stdint.h>

#define FLOAT_TOL 1e-7

#define K 32 //for k-NN.  Do not change!
#define BLOCK_SIZE 16 //must be a power of 2 (current 
// implementation of findRange requires a power of 4, in fact)

#define MAX_BS 65535 //max block size (specified by CUDA)
#define SCAN_WIDTH 1024

#define MEM_USED_IN_SCAN(n) ( 2*( (n) + SCAN_WIDTH-1 )/SCAN_WIDTH*sizeof(unint))

//The distance measure that is used.  This macro returns the 
//distance for a single coordinate.
#define DIST(i,j) ( fabs((i)-(j)) )  // L_1
//#define DIST(i,j) ( ( (i)-(j) )*( (i)-(j) ) )  // L_2

// Format that the data is manipulated in:
typedef float real;
#define MAX_REAL FLT_MAX

// To switch to double precision, comment out the above 
// 2 lines and uncomment the following two lines. 

//typedef double real;
//#define MAX_REAL DBL_MAX

//Percentage of device mem to use
#define MEM_USABLE .95

#define DUMMY_IDX UINT_MAX

//Row major indexing
#define IDX(i,j,ld) (((i)*(ld))+(j))

//increase an int to the next multiple of BLOCK_SIZE
#define PAD(i) ( ((i)%BLOCK_SIZE)==0 ? (i):((i)/BLOCK_SIZE)*BLOCK_SIZE+BLOCK_SIZE ) 

//decrease an int to the next multiple of BLOCK_SIZE
#define DPAD(i) ( ((i)%BLOCK_SIZE)==0 ? (i):((i)/BLOCK_SIZE)*BLOCK_SIZE ) 

//#define MAX(i,j) ((i) > (j) ? (i) : (j))
//#define MIN(i,j) ((i) <= (j) ? (i) : (j))
#define MAXi(i,j,k,l) ((i) > (j) ? (k) : (l)) //indexed version
#define MINi(i,j,k,l) ((i) <= (j) ? (k) : (l))


typedef unsigned __int32 unint;

typedef struct {
  real *mat;
  unint r; //rows
  unint c; //cols
  unint pr; //padded rows
  unint pc; //padded cols
  unint ld; //the leading dimension (in this code, this is the same as pc)
} matrix;


typedef struct {
  char *mat;
  unint r;
  unint c;
  unint pr;
  unint pc;
  unint ld;
} charMatrix;


typedef struct {
  unint *mat;
  unint r;
  unint c;
  unint pr;
  unint pc;
  unint ld;
} intMatrix;


typedef struct{
  unint *numGroups; //The number of groups of DB points to be examined.
  unint *groupCountX; //The number of elements in each DB group.
  unint *qToQGroup; //map from query to query group #.
  unint *qGroupToXGroup; //map from query group to DB gruop
  unint ld; //the width of memPos and groupCount (= max over numGroups)
} compPlan;


typedef struct {
  matrix dx;
  intMatrix dxMap;
  matrix dr;
  unint *groupCount;
} rbcStruct;

#endif
