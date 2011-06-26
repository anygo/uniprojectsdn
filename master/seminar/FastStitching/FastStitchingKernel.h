#ifndef FASTSTITCHINGKERNEL_H__
#define FASTSTITCHINGKERNEL_H__

#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <channel_descriptor.h>
#include <cuda_runtime_api.h>
#include <float.h>

// const memory for transformation matrix
__constant__ float dev_matrix[16];


// Texture that holds the input image
//----------------------------------------------------------------------------
texture<float, 2, cudaReadModeElementType> InputImageTexture;


// Division. If division remainder is neq zero then the result is ceiled
//----------------------------------------------------------------------------
#define DivUp(a,b) ((a % b != 0) ? (a/b + 1) : (a/b))


//----------------------------------------------------------------------------
__global__ void
CUDAFindLandmarksKernel(float4* source, float4* target, float4* source_out, float4* target_out, int* indices_source, int* indices_target, int numLandmarks)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	// check and copy source point
	int idx = indices_source[tid];
	float4 src = source[idx];
	if (src.x != src.x)
	{
		indices_source[tid] = -1;
		source_out[tid].x = sqrtf(-1.f);
	} else
	{
		source_out[tid] = src;
	}

	// check and copy target point
	idx = indices_target[tid];
	float4 tgt = target[idx];
	if (tgt.x != tgt.x)
	{
		indices_target[tid] = -1;
		target_out[tid].x = sqrtf(-1.f);
	} else
	{
		target_out[tid] = tgt;
	}
}

__global__ void
CUDAFindNNBFKernel(float4* source, float4* target, int* correspondences, int numLandmarks)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	// check and copy source point
	float4 src = source[tid];
	if(src.x != src.x) 
	{
		correspondences[tid] = -1;
		return;
	} 
	
	float minDist = FLT_MAX;
	int nn = -1;

	for(int i = 0; i < numLandmarks; ++i)
	{
		float4 cur = target[i];
		if(cur.x != cur.x)
			continue;
		float dist = (src.x - cur.x) * (src.x - cur.x) + (src.y - cur.y) * (src.y - cur.y) + (src.z - cur.z) * (src.z - cur.z);
		if(dist < minDist)
		{
			minDist = dist;
			nn = i;
		}
	}
	correspondences[tid] = nn;
}

__global__ void
CUDATransformLandmarksKernel(float4* toBeTransformed)
{
	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	float4 point = toBeTransformed[tid];

	float x = point.x * dev_matrix[0] + point.y * dev_matrix[1] + point.z * dev_matrix[2] + dev_matrix[3];
	float y = point.x * dev_matrix[4] + point.y * dev_matrix[5] + point.z * dev_matrix[6] + dev_matrix[7];
	float z = point.x * dev_matrix[8] + point.y * dev_matrix[9] + point.z * dev_matrix[10] + dev_matrix[11];
	float w = point.x * dev_matrix[12] + point.y * dev_matrix[13] + point.z * dev_matrix[14] + dev_matrix[15];

	float4 transformed = make_float4(x/w, y/w, z/w, 1.f);
	toBeTransformed[tid] = transformed;
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