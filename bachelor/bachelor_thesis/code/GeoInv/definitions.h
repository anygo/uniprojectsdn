#ifndef DEFINITIONS_H
#define DEFINITIONS_H


#define NUM_FEATURES 6

typedef struct FeatureBoundary
{
	double minVal; // minimum (occured) value 
	double maxVal; // maximum (occured) value
	int bins; // number of bins
} FeatureBoundary; 

typedef struct Derivatives
{
	double Rx;
	double Ry;
	double Rxx;
	double Ryy;
	double Rxy;
} Derivatives;

typedef struct LISO
{
	int coordX;
	int coordY;
	double R;
	double G1;
	double Q;

	double kappa;
	double mu;

	double S1;
	double S2;

	// features
	double feature[NUM_FEATURES]; // m_featureHistLPIP[0](0), grad(1), lambda(2), mass(3), centroid(4), gyration(5)
	Derivatives d;

	// for training
	double gamma;
} LISO;


#endif // DEFINITIONS_H