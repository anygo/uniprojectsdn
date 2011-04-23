#ifndef COLOR_TRIPLE_H
#define COLOR_TRIPLE_H


// struct members named as in the paper:
// M1, M2: mean colors of the two distinct regions
// Mp: mean color on the edge path
//
// rect: rectangular, which describes position and size of the
// patch, which was used for computing the mean values of the 3 colors
struct ColorTriple
{
	cv::Rect rect; // upper left pixel of patch in the image
	cv::Vec3d M1;
	double distM1;
	cv::Vec3d M2;
	double distM2;
	cv::Vec3d Mp;
	double distMp;
};
typedef struct ColorTriple ColorTriple;


// normalized version of a color triple
// does not contain any other information than the mean colors
//struct ColorTripleNormalized
//{
//	cv::Vec3d M1;
//	cv::Vec3d M2;
//	cv::Vec3d Mp;
//};
//typedef struct ColorTripleNormalized ColorTripleNormalized;


#endif