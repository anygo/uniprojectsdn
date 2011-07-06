#ifndef defs_H__
#define	defs_H__

#include "cutil_inline.h"

#define MAX_REPRESENTATIVES 1024
#define CUDA_THREADS_PER_BLOCK 128
#define CUDA_BUFFER_SIZE ( CUDA_THREADS_PER_BLOCK )

// we assume that these sizes won't change
#define FRAME_SIZE_X 640
#define FRAME_SIZE_Y 480

// global values for histogram difference computation
#define MAX_RANGE_VAL 65536
#define NUM_BINS_HIST 128
#define STEP_SIZE_HIST 16 // not every pixel is used to compute the histogram

// GLOBAL TIME STATS
static int OVERALL_TIME;
static int LOAD_TIME;
static int CLIP_TIME;
static int ICP_TIME;
static int TRANSFORM_TIME;


// GPU stuff
// Division. If division remainder is neq zero then the result is ceiled
//----------------------------------------------------------------------------
#define DivUp(a,b) ((a % b != 0) ? (a/b + 1) : (a/b))

// rbc specific structure
typedef struct RepGPU
{
	unsigned int index;
	unsigned int nrOfPoints;
	unsigned int* dev_points;
} RepGPU;


#endif // defs_H__
