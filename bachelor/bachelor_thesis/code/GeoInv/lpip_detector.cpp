#include "lpip_detector.h"

#include <iostream>
#include <fstream>


using namespace cv;

LPIPDetector::LPIPDetector(const std::string filename, unsigned int channel, double errorThreshold, int sobelSize, double gamma)
{
	m_errorThreshold = errorThreshold;
	m_sobelSize = sobelSize;
	m_filename = filename;
	m_channel = channel;
	m_gamma = gamma;

	Mat3f allChannels = imread(filename);
	cv::Mat1f channels[3];
	split(allChannels, channels);

	m_coi = channels[channel];

	// "normalizing" (range of values in image: [0:255] -> [0:1])
	m_coi *= (1./255.);

	
	// gammarize it ;)
	for (int i = 0; i < m_coi.rows; i++)
		for (int j = 0; j < m_coi.cols; j++)
			m_coi[i][j] = (float)pow((float)m_coi[i][j], (float)m_gamma);
}


void LPIPDetector::detect(std::string outputpath)
{
	Mat1f workingCopy;
	m_coi.copyTo(workingCopy);	
	GaussianBlur(workingCopy, workingCopy, Size(5,5), 0.5, 0.5);

	// required derivatives for determining the gradient direction
	Mat1f dx, dy, dxx, dyy, dxy;
	Sobel(workingCopy,  dx, CV_32F, 1, 0, m_sobelSize);
	Sobel(workingCopy,  dy, CV_32F, 0, 1, m_sobelSize);
	Sobel(workingCopy, dxx, CV_32F, 2, 0, m_sobelSize);
	Sobel(workingCopy, dyy, CV_32F, 0, 2, m_sobelSize);
	Sobel(workingCopy, dxy, CV_32F, 1, 1, m_sobelSize);


	for (int i = 5; i < workingCopy.rows-5; i++)
	{
		for (int j = 5; j < workingCopy.cols-5; j++)
		{
			// exclude lisos with R == 0 (objective function -> log(R))
			if (workingCopy[i][j] < 1./255.) // R = workingCopy[i][j]
				continue;

			double x = dx[i][j];
			double y = dy[i][j];

			// this makes it faster
			double criterion = sqrt(x*x + y*y);
			if (criterion < 0.1)
				continue;

			// exclude those:
			if (abs(x) < DBL_EPSILON && abs(y) < DBL_EPSILON)
				continue;

			// compute values for struct LISO
			LISO liso;

			double angle = atan ( y / x ) * 180 / CV_PI;
			Mat1f patch = workingCopy(Rect(j-m_sobelSize, i-m_sobelSize, m_sobelSize*2+1, m_sobelSize*2+1));
			getDerivativesAfterRotation(patch, angle+45., liso.d);


			if (abs(liso.d.Rx) < DBL_EPSILON)
				continue;

			liso.kappa      = -liso.d.Ryy / liso.d.Rx;
			liso.feature[2] =  liso.d.Rxx / liso.d.Rx;
			liso.mu         = -liso.d.Rxy / liso.d.Rx;


			double left = ((liso.feature[2] - liso.kappa - 2*liso.mu) - 
				(liso.feature[2] - liso.kappa + 2*liso.mu)) / liso.d.Rx;
			double middle = ((liso.feature[2] - liso.kappa - 2*liso.mu) / liso.d.Rx - 
				(liso.feature[2] + liso.kappa)) / liso.d.Rx;
			double right = ((liso.feature[2] - liso.kappa + 2*liso.mu) - 
				(liso.feature[2] + liso.kappa)) / liso.d.Rx;

			double error = abs(left) + abs(middle) + abs(right);

			if (error < m_errorThreshold)
			{	
				liso.coordX = j;
				liso.coordY = i;
				liso.R = workingCopy[i][j];
				liso.G1 = (liso.feature[2] - liso.kappa - 2*liso.mu) / liso.d.Rx + 
					(liso.feature[2] - liso.kappa + 2*liso.mu) / liso.d.Rx + 
					(liso.feature[2] + liso.kappa) / liso.d.Rx;
				liso.G1 /= 3.; // mean of three

				liso.Q = 1. / (1. - liso.G1*liso.R);

				if (2.*liso.Q + 1 < DBL_EPSILON) 
					continue;

				liso.Q = (sqrt(3.)/(sqrt(3.)-1.)) * (1. - sqrt(1./(2.*liso.Q + 1.)));

				if (liso.Q < 0.0 || liso.Q > 1.5)
					continue;

				liso.feature[0] = error;
				liso.feature[1] = sqrt(liso.d.Rx*liso.d.Rx + liso.d.Ry*liso.d.Ry);
				liso.gamma = m_gamma;

				m_lisoSet.push_back(liso);	
			}
		}
	}

	m_lisoMap = Mat1f::zeros(workingCopy.rows, workingCopy.cols);
	for (int i = 0; i < (int)m_lisoSet.size(); i++)
	{
		LISO cur = m_lisoSet[i];
		m_lisoMap[cur.coordY][cur.coordX] = 1;
	}

	// write all RQ_values to file (could be very large)
	std::stringstream path1;
	path1 << outputpath << "RQ.txt";
	std::ofstream f(path1.str().c_str());
	for (int i = 0; i < (int)m_lisoSet.size(); i++)
	{
		f << m_lisoSet[i].R << " " << m_lisoSet[i].Q << std::endl;
	}
	f.close();
}


// ccw rotation (size of return mat is greater than size of parameter img)
void LPIPDetector::getDerivativesAfterRotation(cv::Mat1f &img, double degree, Derivatives &d)
{			
	Mat1f rotated;
	Mat rotationMatrix = getRotationMatrix2D(Point2f(img.cols/2.f, img.rows/2.f), degree, 1);
	warpAffine(img, rotated, rotationMatrix, img.size(), INTER_LINEAR);

	Mat1f partialDerivation;
	rotated = rotated(Range(1, img.rows-1), Range(1, img.cols-1));

	Sobel(rotated, partialDerivation, CV_32F, 1, 0, m_sobelSize);
	d.Rx = partialDerivation[partialDerivation.rows/2][partialDerivation.cols/2];
	Sobel(rotated, partialDerivation, CV_32F, 0, 1, m_sobelSize);
	d.Ry = partialDerivation[partialDerivation.rows/2][partialDerivation.cols/2];
	Sobel(rotated, partialDerivation, CV_32F, 2, 0, m_sobelSize);
	d.Rxx = partialDerivation[partialDerivation.rows/2][partialDerivation.cols/2];
	Sobel(rotated, partialDerivation, CV_32F, 0, 2, m_sobelSize);
	d.Ryy = partialDerivation[partialDerivation.rows/2][partialDerivation.cols/2];
	Sobel(rotated, partialDerivation, CV_32F, 1, 1, m_sobelSize);
	d.Rxy = partialDerivation[partialDerivation.rows/2][partialDerivation.cols/2];
}
