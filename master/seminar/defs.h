#ifndef defs_H__
#define	defs_H__

typedef struct Point6D
{
	double x, y, z;
	double r, g, b;
} Point6D;

enum ICP_METRIC
{
	LOG_ABSOLUTE_DISTANCE,
	ABSOLUTE_DISTANCE,
	SQUARED_DISTANCE
};


#endif //defs_H__
