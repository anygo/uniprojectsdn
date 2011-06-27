/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef BRUTE_CU
#define BRUTE_CU

#include "utilsGPU.h"
#include "utils.h"
#include "rbc.h"
#include "defsRBC.h"
#include "kernels.h"
#include "kernelWrap.h"
#include "brute.h"
#include <stdio.h>
#include <cuda.h>
//#include<gsl/gsl_sort.h>

void bruteRangeCount(matrix x, matrix q, real *ranges, unint *cnts){
  matrix dx, dq;
  real *dranges;
  unint *dcnts;
  
  copyAndMove(&dx, &x);
  copyAndMove(&dq, &q);

  ( cudaMalloc( (void**)&dranges, q.pr*sizeof(*dranges) ) );
  cudaMemcpy( dranges, ranges, q.r*sizeof(*dranges), cudaMemcpyHostToDevice );

  ( cudaMalloc( (void**)&dcnts, q.pr*sizeof(*dcnts) ) );
  
  rangeCountWrap(dq, dx, dranges, dcnts);
  
  cudaMemcpy(cnts, dcnts, q.r*sizeof(*cnts), cudaMemcpyDeviceToHost );

  cudaFree(dcnts);
  cudaFree(dranges);
  cudaFree(dx.mat);
  cudaFree(dq.mat);
}


void bruteSearch(matrix x, matrix q, unint *NNs){
  real *dMins;
  unint *dMinIDs;
  matrix dx, dq;

  
  dx.r=x.r; dx.pr=x.pr; dx.c=x.c; dx.pc=x.pc; dx.ld=x.ld;
  dq.r=q.r; dq.pr=q.pr; dq.c=q.c; dq.pc=q.pc; dq.ld=q.ld;

  ( cudaMalloc((void**)&dMins, q.pr*sizeof(*dMins)) );
  ( cudaMalloc((void**)&dMinIDs, q.pr*sizeof(*dMinIDs)) );
  ( cudaMalloc((void**)&dx.mat, dx.pr*dx.pc*sizeof(*dx.mat)) );
  ( cudaMalloc((void**)&dq.mat, dq.pr*dq.pc*sizeof(*dq.mat)) );

  cudaMemcpy(dx.mat,x.mat,x.pr*x.pc*sizeof(*dx.mat),cudaMemcpyHostToDevice);
  cudaMemcpy(dq.mat,q.mat,q.pr*q.pc*sizeof(*dq.mat),cudaMemcpyHostToDevice);
  
  nnWrap(dq,dx,dMins,dMinIDs);

  cudaMemcpy(NNs,dMinIDs,dq.r*sizeof(*NNs),cudaMemcpyDeviceToHost);
  
  cudaFree(dMins);
  cudaFree(dMinIDs);
  cudaFree(dx.mat);
  cudaFree(dq.mat);
}


void bruteK(matrix x, matrix q, intMatrix NNs, matrix NNdists){
  matrix dNNdists;
  intMatrix dMinIDs;
  matrix dx, dq;
  
  dx.r=x.r; dx.pr=x.pr; dx.c=x.c; dx.pc=x.pc; dx.ld=x.ld;
  dq.r=q.r; dq.pr=q.pr; dq.c=q.c; dq.pc=q.pc; dq.ld=q.ld;
  dNNdists.r=q.r; dNNdists.pr=q.pr; dNNdists.c=K; dNNdists.pc=K; dNNdists.ld=dNNdists.pc;
  dMinIDs.r=q.r; dMinIDs.pr=q.pr; dMinIDs.c=K; dMinIDs.pc=K; dMinIDs.ld=dMinIDs.pc;

  ( cudaMalloc((void**)&dNNdists.mat, dNNdists.pc*dNNdists.pr*sizeof(*dNNdists.mat)) );
  ( cudaMalloc((void**)&dMinIDs.mat, dMinIDs.pc*dMinIDs.pr*sizeof(*dMinIDs.mat)) );
  ( cudaMalloc((void**)&dx.mat, dx.pr*dx.pc*sizeof(*dx.mat)) );
  ( cudaMalloc((void**)&dq.mat, dq.pr*dq.pc*sizeof(*dq.mat)) );

  cudaMemcpy(dx.mat,x.mat,x.pr*x.pc*sizeof(*dx.mat),cudaMemcpyHostToDevice);
  cudaMemcpy(dq.mat,q.mat,q.pr*q.pc*sizeof(*dq.mat),cudaMemcpyHostToDevice);
  
  knnWrap(dq,dx,dNNdists,dMinIDs);

  cudaMemcpy(NNs.mat,dMinIDs.mat,NNs.pr*NNs.pc*sizeof(*NNs.mat),cudaMemcpyDeviceToHost);
  cudaMemcpy(NNdists.mat,dNNdists.mat,NNdists.pr*NNdists.pc*sizeof(*NNdists.mat),cudaMemcpyDeviceToHost);

  cudaFree(dNNdists.mat);
  cudaFree(dMinIDs.mat);
  cudaFree(dx.mat);
  cudaFree(dq.mat);
}


void bruteCPU(matrix X, matrix Q, unint *NNs){
  real *dtoNNs; 
  real temp;

  unint i, j;

  dtoNNs = (real*)calloc(Q.r,sizeof(*dtoNNs));
  
  for( i=0; i<Q.r; i++ ){
    dtoNNs[i] = MAX_REAL;
    NNs[i] = 0;
    for(j=0; j<X.r; j++ ){
      temp = distVec( Q, X, i, j );
      if( temp < dtoNNs[i]){
	NNs[i] = j;
	dtoNNs[i] = temp;
      }
    }
  }
  
  free(dtoNNs);  
}


//The following method works properly, but requires the GNU scientific
//library.  If you want to use it, uncomment the code, uncomment the include
//above, and adjust the makefile.  It was used for debugging purposes, but
//is not required by anything else.

/* void bruteKCPU(matrix x, matrix q, intMatrix NNs){ */
/*   int i, j; */

/*   float **d; */
/*   d = (float**)calloc(q.pr, sizeof(*d)); */
/*   size_t **t; */
/*   t = (size_t**)calloc(q.pr, sizeof(*t)); */
/*   for( i=0; i<q.pr; i++){ */
/*     d[i] = (float*)calloc(x.pr, sizeof(**d)); */
/*     t[i] = (size_t*)calloc(x.pr, sizeof(**t)); */
/*   } */

/*   for( i=0; i<q.r; i++){ */
/*     for( j=0; j<x.r; j++) */
/*       d[i][j] = distVec( q, x, i, j ); */
/*     gsl_sort_float_index(t[i], d[i], 1, x.r); */
/*     for ( j=0; j<K; j++) */
/*       NNs.mat[IDX( i, j, NNs.ld )] = t[i][j]; */
/*   } */

/*   for( i=0; i<q.pr; i++){ */
/*     free(t[i]); */
/*     free(d[i]); */
/*   } */
/*   free(t); */
/*   free(d); */
/* } */
#endif
