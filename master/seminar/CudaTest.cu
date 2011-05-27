#include "CudaTestKernel.h"

#include <iostream>
#include <stdio.h>
#include <cutil.h>
#include <stdio.h>
#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <channel_descriptor.h>
#include <cuda_runtime_api.h>
#include "CudaContext.h"


extern "C"
void cudaTest() {

	printf("Testing Cuda...");
	kernel <<<1,1>>>();
	printf("Tested Cuda...");
	
}