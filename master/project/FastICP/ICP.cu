#include "ICPKernel.h"
#include "defs.h"
#include <stdio.h>


//----------------------------------------------------------------------------
extern "C"
void CUDATransformPoints3D(float* points, float* m, unsigned int numPts, unsigned int dim)
{
	if (dim == 6)
	{
		if (numPts == 32)
			kernelTransformPoints3D<32,6><<<DivUp(32, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 128)
			kernelTransformPoints3D<128,6><<<DivUp(128, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 256)
			kernelTransformPoints3D<256,6><<<DivUp(256, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 512)
			kernelTransformPoints3D<512,6><<<DivUp(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 1024)
			kernelTransformPoints3D<1024,6><<<DivUp(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 2048)
			kernelTransformPoints3D<2048,6><<<DivUp(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 4096)
			kernelTransformPoints3D<4096,6><<<DivUp(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 8192)
			kernelTransformPoints3D<8192,6><<<DivUp(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == 16384)
			kernelTransformPoints3D<16384,6><<<DivUp(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
		else if (numPts == KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT)
			kernelTransformPoints3D<KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT,6><<<DivUp(KINECT_IMAGE_WIDTH*KINECT_IMAGE_HEIGHT, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, m);
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
void CUDAComputeCentroid(float* points, float* out, unsigned long* correspondences, unsigned int numPts, unsigned int dim)
{
	if (dim == 3)
	{
		if (numPts <= 1)
			kernelComputeCentroid<1,3><<<DivUp(1, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 2)
			kernelComputeCentroid<2,3><<<DivUp(2, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 4)
			kernelComputeCentroid<4,3><<<DivUp(4, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 8)
			kernelComputeCentroid<8,3><<<DivUp(8, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 16)
			kernelComputeCentroid<16,3><<<DivUp(16, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 32)
			kernelComputeCentroid<32,3><<<DivUp(32, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 64)
			kernelComputeCentroid<64,3><<<DivUp(64, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 128)
			kernelComputeCentroid<128,3><<<DivUp(128, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 256)
			kernelComputeCentroid<256,3><<<DivUp(256, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 512)
			kernelComputeCentroid<512,3><<<DivUp(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 1024)
			kernelComputeCentroid<1024,3><<<DivUp(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 2048)
			kernelComputeCentroid<2048,3><<<DivUp(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 4096)
			kernelComputeCentroid<4096,3><<<DivUp(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 8192)
			kernelComputeCentroid<8192,3><<<DivUp(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 16384)
			kernelComputeCentroid<16384,3><<<DivUp(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else 
			printf("[%s] no instance for (numPts,dim)=(%d,%d) available\n", __FUNCTION__, numPts, dim);
	}
	else if (dim == 6)
	{
		if (numPts <= 1)
			kernelComputeCentroid<1,6><<<DivUp(1, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 2)
			kernelComputeCentroid<2,6><<<DivUp(2, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 4)
			kernelComputeCentroid<4,6><<<DivUp(4, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 8)
			kernelComputeCentroid<8,6><<<DivUp(8, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 16)
			kernelComputeCentroid<16,6><<<DivUp(16, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 32)
			kernelComputeCentroid<32,6><<<DivUp(32, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 64)
			kernelComputeCentroid<64,6><<<DivUp(64, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 128)
			kernelComputeCentroid<128,6><<<DivUp(128, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 256)
			kernelComputeCentroid<256,6><<<DivUp(256, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 512)
			kernelComputeCentroid<512,6><<<DivUp(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 1024)
			kernelComputeCentroid<1024,6><<<DivUp(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 2048)
			kernelComputeCentroid<2048,6><<<DivUp(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 4096)
			kernelComputeCentroid<4096,6><<<DivUp(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 8192)
			kernelComputeCentroid<8192,6><<<DivUp(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
		else if (numPts == 16384)
			kernelComputeCentroid<16384,6><<<DivUp(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(points, out, correspondences);
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
			kernelBuildMMatrices<32,6><<<DivUp(32, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 128)
			kernelBuildMMatrices<128,6><<<DivUp(128, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 256)
			kernelBuildMMatrices<256,6><<<DivUp(256, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 512)
			kernelBuildMMatrices<512,6><<<DivUp(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 1024)
			kernelBuildMMatrices<1024,6><<<DivUp(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 2048)
			kernelBuildMMatrices<2048,6><<<DivUp(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 4096)
			kernelBuildMMatrices<4096,6><<<DivUp(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 8192)
			kernelBuildMMatrices<8192,6><<<DivUp(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
		else if (numPts == 16384)
			kernelBuildMMatrices<16384,6><<<DivUp(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(moving, fixed, centroidMoving, centroidFixed, out, correspondences);
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
		kernelReduceMMatrices<1><<<DivUp(1, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 2)
		kernelReduceMMatrices<2><<<DivUp(2, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 4)
		kernelReduceMMatrices<4><<<DivUp(4, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 8)
		kernelReduceMMatrices<8><<<DivUp(8, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 16)
		kernelReduceMMatrices<16><<<DivUp(16, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 32)
		kernelReduceMMatrices<32><<<DivUp(32, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 64)
		kernelReduceMMatrices<64><<<DivUp(64, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 128)
		kernelReduceMMatrices<128><<<DivUp(128, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 256)
		kernelReduceMMatrices<256><<<DivUp(256, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 512)
		kernelReduceMMatrices<512><<<DivUp(512, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 1024)
		kernelReduceMMatrices<1024><<<DivUp(1024, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 2048)
		kernelReduceMMatrices<2048><<<DivUp(2048, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 4096)
		kernelReduceMMatrices<4096><<<DivUp(4096, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 8192)
		kernelReduceMMatrices<8192><<<DivUp(8192, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else if (numPts == 16384)
		kernelReduceMMatrices<16384><<<DivUp(16384, CUDA_THREADS_PER_BLOCK), CUDA_THREADS_PER_BLOCK>>>(matrices, out);
	else 
		printf("[%s] no instance for numPts=%d available\n", __FUNCTION__, numPts);
}