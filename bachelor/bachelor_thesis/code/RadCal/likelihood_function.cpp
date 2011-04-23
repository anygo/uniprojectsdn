#include "likelihood_function.h"
#include <fstream>
#include <iostream>


using namespace cv;

// constructor reads H and meanCurve (g0) from given filenames
LikelihoodFunction::LikelihoodFunction(const char *filenameH, const char *filenameMeanCurve, double lambda)
{
	m_H = readMatrix(filenameH);
	m_g0 = readMatrix(filenameMeanCurve);
	m_lambda = lambda;
}


// computes (usually 3) inverse response functions with given coefficients
// see equation 8
cv::Mat1f LikelihoodFunction::computeFunctions(cv::Mat1f &coefficients)
{
	Mat1f functions(coefficients.rows, m_H.cols);
	for (int i = 0; i < functions.rows; i++)
	{
		functions.row(i) = m_g0 + coefficients.row(i) * m_H.rowRange(0, coefficients.cols);
	}
	return functions;
}

// computes total distance given by a specific inverse response function triple
// -> equation 7
double LikelihoodFunction::getTotalDistance(const std::vector<ColorTriple> &omega, cv::Mat1f &coefficients)
{
	double totalDistance = 0;

	Mat1f functionTriple = computeFunctions(coefficients);

	for (int i = 0; i < static_cast<int>(omega.size()); i++)
	{
		ColorTriple cur = omega.at(i);
		cv::Vec3d g_M1(
			functionTriple[0][static_cast<int>(cur.M1.val[0]*1023)],
			functionTriple[1][static_cast<int>(cur.M1.val[1]*1023)],
			functionTriple[2][static_cast<int>(cur.M1.val[2]*1023)]
			);
		cv::Vec3d g_Mp(
			functionTriple[0][static_cast<int>(cur.Mp.val[0]*1023)],
			functionTriple[1][static_cast<int>(cur.Mp.val[1]*1023)],
			functionTriple[2][static_cast<int>(cur.Mp.val[2]*1023)]
			);
		cv::Vec3d g_M2(
			functionTriple[0][static_cast<int>(cur.M2.val[0]*1023)],
			functionTriple[1][static_cast<int>(cur.M2.val[1]*1023)],
			functionTriple[2][static_cast<int>(cur.M2.val[2]*1023)]
			);

			Vec3d crossP = (g_M1 - g_M2).cross(g_M1 - g_Mp);
			double distance = norm(crossP) / norm(g_M1-g_M2);
			totalDistance += distance;
	}

	double totalDistanceMean = (omega.size() != 0) ? totalDistance / static_cast<double>(omega.size()) : totalDistance;

	return totalDistanceMean;
}


// see equation 10, for optimization use equation 11, not this
double LikelihoodFunction::likelihoodFunction(const std::vector<ColorTriple> &omega, cv::Mat1f &coefficients)
{
	double expTerm = -m_lambda * getTotalDistance(omega, coefficients);
	double ret = exp(expTerm) / omega.size(); //TODO normalization constant?!?!.. check that!
	return ret;
}


// reads matrix (type float) from file
cv::Mat1f LikelihoodFunction::readMatrix(const char *filename)
{
	int rows, cols;

	std::ifstream file(filename);
	if (!file.is_open())
	{
		std::cout << "could not open " << filename << std::endl;
	}

	file >> rows; 
	file >> cols;

	Mat1f m(rows, cols);
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			file >> m[i][j];
		}
	}

	file.close();

	return m;
}
