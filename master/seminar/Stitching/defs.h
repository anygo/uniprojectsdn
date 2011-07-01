#ifndef defs_H__
#define	defs_H__

#include <cutil_inline.h>

#define MAX_REPRESENTATIVES 1024
#define CUDA_THREADS_PER_BLOCK 128
#define CUDA_BUFFER_SIZE ( CUDA_THREADS_PER_BLOCK )

#define SizeX 640
#define SizeY 480

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

// structure that holds the spatial information of a single point
typedef struct PointCoords
{
	float x, y, z;
}
PointCoords;

// structure that holds the color information of a single point
typedef struct PointColors
{
	float r, g, b;
}
PointColors;

// gpu config (avoid too many parameters in gpu-code)
typedef struct GPUConfig
{
	float weightRGB;
	int metric;
	int nrOfPoints;
	PointCoords* targetCoords;
	PointColors* targetColors;
	PointCoords* sourceCoords;
	PointColors* sourceColors;
	unsigned short* indices;
	float* distances;
} GPUConfig;

// rbc specific structure
typedef struct RepGPU
{
	PointCoords coords;
	PointColors colors;
	unsigned short nrOfPoints;
	unsigned short* dev_points;
} RepGPU;


// enum that includes all implemented distance metrics used during ICP
enum ICP_METRIC
{
	LOG_ABSOLUTE_DISTANCE,
	ABSOLUTE_DISTANCE,
	SQUARED_DISTANCE
};


#endif // defs_H__
