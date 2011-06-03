#ifndef defs_H__
#define	defs_H__

typedef struct PointCoords
{
	float x, y, z;
}
PointCoords;

typedef struct PointColors
{
	short r, g, b;
}
PointColors;

enum ICP_METRIC
{
	LOG_ABSOLUTE_DISTANCE,
	ABSOLUTE_DISTANCE,
	SQUARED_DISTANCE
};


#endif // defs_H__
