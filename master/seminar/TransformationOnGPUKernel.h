#ifndef TransformationOnGPUKernel_H__
#define TransformationOnGPUKernel_H__

////////////////////////
// CURRENTLY NOT USED //
////////////////////////

#include "defs.h"

#include <cutil_inline.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <channel_descriptor.h>
#include <cuda_runtime_api.h>

__global__
void kernelTransformPoints(PointCoords* coords, float* m)
{
	// get source[tid] for this thread
	unsigned int tid = blockIdx.x;

	// compute homogeneous transformation
	float x = m[0]*coords[tid].x + m[1]*coords[tid].y + m[2]*coords[tid].z + m[3];
	float y = m[4]*coords[tid].x + m[5]*coords[tid].y + m[6]*coords[tid].z + m[7];
	float z = m[8]*coords[tid].x + m[9]*coords[tid].y + m[10]*coords[tid].z + m[11];
	float w = m[12]*coords[tid].x + m[13]*coords[tid].y + m[14]*coords[tid].z + m[15];

	// set new coordinates
	coords[tid].x = x/w;
	coords[tid].y = y/w;
	coords[tid].z = z/w;
}

#endif // TransformationOnGPUKernel_H__