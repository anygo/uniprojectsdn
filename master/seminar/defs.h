#ifndef defs_H__
#define	defs_H__

#define MAX_REPRESENTATIVES 1024

// structure that holds the spatial information of a single point
typedef struct PointCoords
{
	float x, y, z;
}
PointCoords;

// structure that holds the color information of a single point
typedef struct PointColors
{
	short r, g, b;
}
PointColors;

// rbc specific structure
typedef struct RepGPU
{
	unsigned short index;
	unsigned short nrOfPoints;
	unsigned short* points;
	unsigned short* dev_points;
} RepGPU;

// enum that includes all implemented distance metrics used during ICP
enum ICP_METRIC
{
	LOG_ABSOLUTE_DISTANCE,
	ABSOLUTE_DISTANCE,
	SQUARED_DISTANCE
};

// debug macro
//#define DEBUG

#ifdef DEBUG
#define DBG ( std::cout )
#else
#define DBG if (false) ( std::cout )
#endif


#endif // defs_H__
