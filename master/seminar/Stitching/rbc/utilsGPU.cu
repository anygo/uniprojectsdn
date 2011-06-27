/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef UTILSGPU_CU
#define UTILSGPU_CU

#include <cuda.h>
#include <stdio.h>
#include "defsRBC.h"
#include "utilsGPU.h"

void copyAndMove(matrix *dx, const matrix *x){
  dx->r = x->r; 
  dx->c = x->c;
  dx->pr = x->pr;
  dx->pc = x->pc;
  dx->ld = x->ld;

  ( cudaMalloc( (void**)&(dx->mat), dx->pr*dx->pc*sizeof(*(dx->mat)) ) );
  cudaMemcpy( dx->mat, x->mat, dx->pr*dx->pc*sizeof(*(dx->mat)), cudaMemcpyHostToDevice );
  
}


void copyAndMoveI(intMatrix *dx, const intMatrix *x){
  dx->r = x->r; 
  dx->c = x->c;
  dx->pr = x->pr;
  dx->pc = x->pc;
  dx->ld = x->ld;

  ( cudaMalloc( (void**)&(dx->mat), dx->pr*dx->pc*sizeof(*(dx->mat)) ) );
  cudaMemcpy( dx->mat, x->mat, dx->pr*dx->pc*sizeof(*(dx->mat)), cudaMemcpyHostToDevice );
  
}


void copyAndMoveC(charMatrix *dx, const charMatrix *x){
  dx->r = x->r; 
  dx->c = x->c;
  dx->pr = x->pr;
  dx->pc = x->pc;
  dx->ld = x->ld;

  ( cudaMalloc( (void**)&(dx->mat), dx->pr*dx->pc*sizeof(*(dx->mat)) ) );
  cudaMemcpy( dx->mat, x->mat, dx->pr*dx->pc*sizeof(*(dx->mat)), cudaMemcpyHostToDevice );
  
}



#endif
