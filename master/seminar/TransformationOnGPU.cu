#include "TransformationOnGPUKernel.h"
#include <cutil_inline.h>

////////////////////////
// CURRENTLY NOT USED //
////////////////////////


extern "C"
void TransformationOnGPU(double transformationMatrix[4][4], PointCoords* host_coords)
{
	int nrOfPoints = sizeof(host_coords)/sizeof(PointCoords);

	float* dev_transformationMatrix;
	PointCoords* dev_coords;

	// allocate memory for transformation matrix (will be stored linearly) and copy it
	float m[16];
	m[0] = (float)transformationMatrix[0][0];
	m[1] = (float)transformationMatrix[0][1];
	m[2] = (float)transformationMatrix[0][2];
	m[3] = (float)transformationMatrix[0][3];
	m[4] = (float)transformationMatrix[1][0];
	m[5] = (float)transformationMatrix[1][1];
	m[6] = (float)transformationMatrix[1][2];
	m[7] = (float)transformationMatrix[1][3];
	m[8] = (float)transformationMatrix[2][0];
	m[9] = (float)transformationMatrix[2][1];
	m[10] = (float)transformationMatrix[2][2];
	m[11] = (float)transformationMatrix[2][3];
	m[12] = (float)transformationMatrix[3][0];
	m[13] = (float)transformationMatrix[3][1];
	m[14] = (float)transformationMatrix[3][2];
	m[15] = (float)transformationMatrix[3][3];
	
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_transformationMatrix, 16*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpy(dev_transformationMatrix, m, 16*sizeof(float), cudaMemcpyHostToDevice));
	
	// allocate memory for coords on gpu and copy it
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_coords, nrOfPoints*sizeof(PointCoords)));
	CUDA_SAFE_CALL(cudaMemcpy(dev_coords, host_coords, nrOfPoints*sizeof(PointCoords), cudaMemcpyHostToDevice));
	
	// compute transformations
	kernelTransformPoints<<<nrOfPoints,1>>>(dev_coords, dev_transformationMatrix);
	CUT_CHECK_ERROR("Kernel execution failed (while transforming points)");
	
	// copy transformed points back to host
	CUDA_SAFE_CALL(cudaMemcpy(host_coords, dev_coords, nrOfPoints*sizeof(PointCoords), cudaMemcpyDeviceToHost));
	
	// cleanup gpu
	CUDA_SAFE_CALL(cudaFree(dev_coords));
	CUDA_SAFE_CALL(cudaFree(dev_transformationMatrix));
}