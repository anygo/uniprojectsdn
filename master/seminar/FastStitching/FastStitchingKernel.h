#ifndef FASTSTITCHINGKERNEL_H__
#define FASTSTITCHINGKERNEL_H__

#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <channel_descriptor.h>
#include <cuda_runtime_api.h>


// Texture that holds the input image
//----------------------------------------------------------------------------
texture<float, 2, cudaReadModeElementType> InputImageTexture;


// Division. If division remainder is neq zero then the result is ceiled
//----------------------------------------------------------------------------
#define DivUp(a,b) ((a % b != 0) ? (a/b + 1) : (a/b))

//----------------------------------------------------------------------------
__global__ void
CUDANearestNeighborBFKernel(unsigned int NX, unsigned int NY, float4* source, float4* target, unsigned int* corr )
{

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	// 2D index and linear index within this thread block
	int tu = threadIdx.x;
	int tv = threadIdx.y;

	// Global 2D index and linear index.
//	float gu = blockIdx.x*BlockSizeX+tu;
//	float gv = blockIdx.y*BlockSizeY+tv;

	// Check for out-of-bounds


	// DO Calculate the Correspondences...

}


//----------------------------------------------------------------------------
template<unsigned int BlockSizeX, unsigned int BlockSizeY>
__global__ void
CUDARangeToWorldKernel(unsigned int NX, unsigned int NY, float4* Output, float4* duplicate, float fx, float fy, float cx, float cy, float k1, float k2)
{
	// 2D index and linear index within this thread block
	int tu = threadIdx.x;
	int tv = threadIdx.y;

	// Global 2D index and linear index.
	float gu = blockIdx.x*BlockSizeX+tu;
	float gv = blockIdx.y*BlockSizeY+tv;

	// Check for out-of-bounds
	if ( gu >= NX || gv >= NY )
		return;

	// The range value
	float value = tex2D(InputImageTexture, gu, gv);

	// The corresponding x,y,z triple
	float4 WC;

	if ( value < 500.f )
		value = sqrtf(-1.0f);

	float X2Z = 1.209f;
	float Y2Z = 0.9132f;
	float fNormalizedX = gu / NX - 0.5f; // check for float
	float x = fNormalizedX * value * X2Z;

	float fNormalizedY = 0.5f - gv / NY;
	float y = fNormalizedY * value * Y2Z;

	// World coordinates
	WC = make_float4(x, y, value, 1.0f);

	
	// Set the WC for the duplicate without Mesh Structure
	duplicate[(int)(gv*NX + gu)] = WC;

	// Mesh
	// the size of the outputImg is twice the size of the input because one line does not only
	// represent the points of one line but the triangles of one strip
	int oNX = 2*NX;
	int ou = 2*gu;
	if ( gv != NY-1 )
	{
		Output[(int)(gv*oNX+ou)] = WC;
	}
	if ( gv != 0 )
	{
		Output[(int)((gv-1)*oNX+ou+1)] = WC;
	}

}


#endif // FASTSTITCHINGKERNEL_H__