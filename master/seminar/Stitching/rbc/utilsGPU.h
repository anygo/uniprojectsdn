/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef UTILSGPU_H
#define UTILSGPU_H

#include "defsRBC.h"
#include <cuda.h>


void copyAndMove(matrix*,const matrix*);
void copyAndMoveI(intMatrix*,const intMatrix*);
void copyAndMoveC(charMatrix*,const charMatrix*);

#endif
