#include "ICPKernel.h"
#include "defs.h"
#include "ritkCudaMacros.h"
#include <stdio.h>


//----------------------------------------------------------------------------
extern "C"
void CUDATransformPoints3D(float* points, float* m, unsigned int numPts, unsigned int dim)
{
	if (dim == 6)
	{
		if (numPts == 512)
			kernelTransformPoints3D<512,6><<<DIVUP(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 1024)
			kernelTransformPoints3D<1024,6><<<DIVUP(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 2048)
			kernelTransformPoints3D<2048,6><<<DIVUP(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 4096)
			kernelTransformPoints3D<4096,6><<<DIVUP(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 8192)
			kernelTransformPoints3D<8192,6><<<DIVUP(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 16384)
			kernelTransformPoints3D<16384,6><<<DIVUP(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT)
			kernelTransformPoints3D<KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT,6><<<DIVUP(KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else 
			printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
	else {
		printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
}


//----------------------------------------------------------------------------
extern "C"
void CUDAAccumulateMatrix(float* accu, float* m)
{
	kernelAccumulateMatrix<<<1, 16>>>(accu, m);
}


//----------------------------------------------------------------------------
extern "C"
void CUDAComputeCentroid3D(float* points, float* out, unsigned long* correspondences, unsigned int numPts, unsigned int dim)
{
	if (dim == 3)
	{
		if (numPts <= 1)
			kernelComputeCentroid<1,3,1><<<1, 1>>>(points, out, correspondences);
		else if (numPts == 2)
			kernelComputeCentroid<2,3,2><<<1, 2>>>(points, out, correspondences);
		else if (numPts == 4)
			kernelComputeCentroid<4,3,4><<<1, 4>>>(points, out, correspondences);
		else if (numPts == 8)
			kernelComputeCentroid<8,3,8><<<1, 8>>>(points, out, correspondences);
		else if (numPts == 16)
			kernelComputeCentroid<16,3,16><<<1, 16>>>(points, out, correspondences);
		else if (numPts == 32)
			kernelComputeCentroid<32,3,32><<<1, 32>>>(points, out, correspondences);
		else if (numPts == 64)
			kernelComputeCentroid<64,3,64><<<1, 64>>>(points, out, correspondences);
		else if (numPts == 128)
			kernelComputeCentroid<128,3,CUDA_THREADS_PER_BLOCK><<<DIVUP(128, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 256)
			kernelComputeCentroid<256,3,CUDA_THREADS_PER_BLOCK><<<DIVUP(256, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 512)
			kernelComputeCentroid<512,3,CUDA_THREADS_PER_BLOCK><<<DIVUP(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 1024)
			kernelComputeCentroid<1024,3,CUDA_THREADS_PER_BLOCK><<<DIVUP(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 2048)
			kernelComputeCentroid<2048,3,CUDA_THREADS_PER_BLOCK><<<DIVUP(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 4096)
			kernelComputeCentroid<4096,3,CUDA_THREADS_PER_BLOCK><<<DIVUP(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 8192)
			kernelComputeCentroid<8192,3,CUDA_THREADS_PER_BLOCK><<<DIVUP(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 16384)
			kernelComputeCentroid<16384,3,CUDA_THREADS_PER_BLOCK><<<DIVUP(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else 
			printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
	else if (dim == 6)
	{
		if (numPts <= 1)
			kernelComputeCentroid<1,6,1><<<1, 1>>>(points, out, correspondences);
		else if (numPts == 2)
			kernelComputeCentroid<2,6,2><<<1, 2>>>(points, out, correspondences);
		else if (numPts == 4)
			kernelComputeCentroid<4,6,4><<<1, 4>>>(points, out, correspondences);
		else if (numPts == 8)
			kernelComputeCentroid<8,6,8><<<1, 8>>>(points, out, correspondences);
		else if (numPts == 16)
			kernelComputeCentroid<16,6,16><<<1, 16>>>(points, out, correspondences);
		else if (numPts == 32)
			kernelComputeCentroid<32,6,32><<<1, 32>>>(points, out, correspondences);
		else if (numPts == 64)
			kernelComputeCentroid<64,6,64><<<1, 64>>>(points, out, correspondences);
		else if (numPts == 128)
			kernelComputeCentroid<128,6,CUDA_THREADS_PER_BLOCK><<<DIVUP(128, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 256)
			kernelComputeCentroid<256,6,CUDA_THREADS_PER_BLOCK><<<DIVUP(256, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 512)
			kernelComputeCentroid<512,6,CUDA_THREADS_PER_BLOCK><<<DIVUP(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 1024)
			kernelComputeCentroid<1024,6,CUDA_THREADS_PER_BLOCK><<<DIVUP(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 2048)
			kernelComputeCentroid<2048,6,CUDA_THREADS_PER_BLOCK><<<DIVUP(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 4096)
			kernelComputeCentroid<4096,6,CUDA_THREADS_PER_BLOCK><<<DIVUP(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 8192)
			kernelComputeCentroid<8192,6,CUDA_THREADS_PER_BLOCK><<<DIVUP(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 16384)
			kernelComputeCentroid<16384,6,CUDA_THREADS_PER_BLOCK><<<DIVUP(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else 
			printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
	else {
		printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
}


//----------------------------------------------------------------------------
extern "C"
void CUDABuildMMatrices(float* moving, float* fixed, float* centroidMoving, float* centroidFixed, float* out, unsigned long* correspondences, unsigned int numPts, unsigned int dim)
{
	if (dim == 6)
	{
		if (numPts == 32)
			kernelBuildMMatrices<32,6><<<DIVUP(32, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 128)
			kernelBuildMMatrices<128,6><<<DIVUP(128, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 256)
			kernelBuildMMatrices<256,6><<<DIVUP(256, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 512)
			kernelBuildMMatrices<512,6><<<DIVUP(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 1024)
			kernelBuildMMatrices<1024,6><<<DIVUP(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 2048)
			kernelBuildMMatrices<2048,6><<<DIVUP(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 4096)
			kernelBuildMMatrices<4096,6><<<DIVUP(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 8192)
			kernelBuildMMatrices<8192,6><<<DIVUP(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 16384)
			kernelBuildMMatrices<16384,6><<<DIVUP(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else 
			printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
	else {
		printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
}


//----------------------------------------------------------------------------
extern "C"
void CUDAReduceMMatrices(float* matrices, float* out, unsigned int numPts)
{
	if (numPts <= 1)
		kernelReduceMMatrices<1><<<DIVUP(1, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 2)
		kernelReduceMMatrices<2><<<DIVUP(2, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 4)
		kernelReduceMMatrices<4><<<DIVUP(4, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 8)
		kernelReduceMMatrices<8><<<DIVUP(8, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 16)
		kernelReduceMMatrices<16><<<DIVUP(16, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 32)
		kernelReduceMMatrices<32><<<DIVUP(32, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 64)
		kernelReduceMMatrices<64><<<DIVUP(64, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 128)
		kernelReduceMMatrices<128><<<DIVUP(128, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 256)
		kernelReduceMMatrices<256><<<DIVUP(256, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 512)
		kernelReduceMMatrices<512><<<DIVUP(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 1024)
		kernelReduceMMatrices<1024><<<DIVUP(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 2048)
		kernelReduceMMatrices<2048><<<DIVUP(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 4096)
		kernelReduceMMatrices<4096><<<DIVUP(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 8192)
		kernelReduceMMatrices<8192><<<DIVUP(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 16384)
		kernelReduceMMatrices<16384><<<DIVUP(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else 
		printf("[%s] no instance for numPts=%d available\n", __FUNCTION__, numPts);
}


//----------------------------------------------------------------------------
extern "C"
void CUDAEstimateTransformationFromMMatrix(float* centroidMoving, float* centroidFixed, float* matrix, float* outMatrix)
{
	kernelEstimateTransformationFromMMatrix<<<1, 1>>>(centroidMoving, centroidFixed, matrix, outMatrix);
}
