#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
#endif

#include "FastStitchingKernel.h"
#include <cutil.h>

//#define RUNTIME_EVALUATION

//----------------------------------------------------------------------------
extern "C"
float
CUDAMeshTriangulation(unsigned int w, unsigned int h, const float3* Input, float4* Output)
{
	float ElapsedTime = -1;

	// Timer for runtime evaluation
#ifdef RUNTIME_EVALUATION
	unsigned int NumRuntimeIterations = 1000;
	unsigned int Timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&Timer));
	CUT_SAFE_CALL(cutStartTimer(Timer));
	for ( unsigned int i = 0; i < NumRuntimeIterations; i++ ) 
	{
#endif
	
	// Kernel Invocation
	dim3 DimBlock(16,16);
	dim3 DimGrid(DivUp(w,DimBlock.x),DivUp(h,DimBlock.y));
	CUDAMeshTriangulationKernel<16,16><<<DimGrid,DimBlock>>>(w, h, Input, Output);

#ifdef RUNTIME_EVALUATION
	}
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(Timer));
	//printf("	Runtime for CUDA Range2World: %fms \n\n",(cutGetTimerValue(Timer)/NumRuntimeIterations));
	ElapsedTime = cutGetTimerValue(Timer)/NumRuntimeIterations;
	CUT_SAFE_CALL(cutDeleteTimer(Timer));

#endif

	CUT_CHECK_ERROR("Kernel execution failed");

	return ElapsedTime;
}


//----------------------------------------------------------------------------
extern "C"
float
FastStitching(const cudaArray *InputImageArray, float4 *DeviceOutput, int w, int h, float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2)
{
	float ElapsedTime = -1;

	// Set input image texture parameters and bind texture to the array. Texture is defined in the kernel
	InputImageTexture.addressMode[0] = cudaAddressModeClamp;
	InputImageTexture.addressMode[1] = cudaAddressModeClamp;
	InputImageTexture.filterMode = cudaFilterModePoint;
	InputImageTexture.normalized = false;
	cutilSafeCall(cudaBindTextureToArray(InputImageTexture, InputImageArray));

	// Timer for runtime evaluation
#ifdef RUNTIME_EVALUATION
	unsigned int NumRuntimeIterations = 1000;
	unsigned int Timer = 0;
	CUT_SAFE_CALL(cutCreateTimer(&Timer));
	CUT_SAFE_CALL(cutStartTimer(Timer));
	for ( unsigned int i = 0; i < NumRuntimeIterations; i++ ) 
	{
#endif
	
	// Kernel Invocation
	dim3 DimBlock(16,16);
	dim3 DimGrid(DivUp(w,DimBlock.x),DivUp(h,DimBlock.y));
	FastStitchingKernel<16,16><<<DimGrid,DimBlock>>>(w,h,DeviceOutput, fx, fy, cx, cy, k1, k2, p1, p2);

#ifdef RUNTIME_EVALUATION
	}
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(Timer));
	//printf("	Runtime for CUDA Range2World: %fms \n\n",(cutGetTimerValue(Timer)/NumRuntimeIterations));
	ElapsedTime = cutGetTimerValue(Timer)/NumRuntimeIterations;
	CUT_SAFE_CALL(cutDeleteTimer(Timer));

#endif

	// Unbind texture
	cutilSafeCall(cudaUnbindTexture(InputImageTexture));

	CUT_CHECK_ERROR("Kernel execution failed");

	return ElapsedTime;
}
