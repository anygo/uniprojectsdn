 /*
  * =============================================================
  * timestwoalt.c - example found in API guide
  *
  * Use mxGetScalar to return the values of scalars instead of 
  * pointers to copies of scalar variables.
  *
  * This is a MEX-file for MATLAB.
  * Copyright (c) 1984-2000 The MathWorks, Inc.
  * =============================================================
  */
  
 /* $Revision: 1.1.4.9 $ */
 
 #include "mex.h"
 
 void timestwoalt(double *y, double x)
 {
   *y = 2.0*x;
 }
 
 