#include <iostream>

__global__ void kernel() { }

void cudaTest() {

	printf("Testing Cuda...");
	kernel<<1,1>>();
	printf("Tested Cuda...");
	
	
}