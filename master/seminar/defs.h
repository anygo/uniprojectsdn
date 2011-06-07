#ifndef defs_H__
#define	defs_H__

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

// enum that includes all implemented distance metrics used during ICP
enum ICP_METRIC
{
	LOG_ABSOLUTE_DISTANCE,
	ABSOLUTE_DISTANCE,
	SQUARED_DISTANCE
};


#endif // defs_H__
