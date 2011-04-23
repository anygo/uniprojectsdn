#include "trainer.h"
#include <iostream>
#include <math.h>
#include <fstream>


using namespace cv;

// constructor
Trainer::Trainer()
{
	reset();
}


// resets everything
void Trainer::reset()
{
	for (int i = 0; i < NUM_FEATURES; i++)
	{
		m_featureBoundaries[i].minVal = DBL_MAX;
		m_featureBoundaries[i].maxVal = DBL_MIN;
	}

	int nBins[NUM_FEATURES] = {100, 100, 100, 25, 100, 100};
	setNumBins(nBins);

	m_lpip.clear();
	m_nonlpip.clear();

	m_histComputed = false;
}


// set number of bins for hist
void Trainer::setNumBins(int binsPerFeature[NUM_FEATURES])
{
	for (int i = 0; i < NUM_FEATURES; i++)
	{
		m_featureBoundaries[i].bins = binsPerFeature[i];
	}
}	

// computes the six features
void Trainer::computeFeatures(std::vector<LISO>& lisoSet, cv::Mat1f& lisoMap)
{
	std::vector<cv::Vec3d> m012;

	for (int i = 0; i < (int) lisoSet.size(); i++)
	{
		LISO *liso = &lisoSet[i];

		int xFrom = liso->coordX-2;
		int xTo   = liso->coordX+2;
		int yFrom = liso->coordY-2;
		int yTo   = liso->coordY+2;

		// compute total mass (m0)
		liso->feature[3] = 0;
		Mat1d b = Mat1d::zeros(5, 5); // b(x, y) -> b[y][x]
		for (int x = xFrom; x <= xTo; x++)
		{
			for (int y = yFrom; y <= yTo; y++)
			{
				b[y-yFrom][x-xFrom] = lisoMap[y][x];
				liso->feature[3] += lisoMap[y][x];
			}
		}

		// compute centroid (m1) and radius of gyration (m2)
		double m1x = 0, m2x = 0, m1y = 0, m2y = 0; // equation 15
		for (int x = 0; x < 5; x++)
		{
			for (int y = 0; y < 5; y++)
			{
				if (b[y][x] != 0) // factor b(x, y)
				{
					m1x += x;
					m2x += x*x;
					m1y += y;
					m2y += y*y;
				}
			}
		}
		m1x /= liso->feature[3];
		m2x /= liso->feature[3];
		m1y /= liso->feature[3];
		m2y /= liso->feature[3];

		liso->feature[4] = sqrt(m1x*m1x + m1y*m1y);
		liso->feature[5] = sqrt(m2x + m2y);

		// m1
		if (liso->feature[4] > m_featureBoundaries[4].maxVal)
			m_featureBoundaries[4].maxVal = liso->feature[4];
		if (liso->feature[4] < m_featureBoundaries[4].minVal)
			m_featureBoundaries[4].minVal = liso->feature[4];
		// m2
		if (liso->feature[5] > m_featureBoundaries[5].maxVal)
			m_featureBoundaries[5].maxVal = liso->feature[5];
		if (liso->feature[5] < m_featureBoundaries[5].minVal)
			m_featureBoundaries[5].minVal = liso->feature[5];
		// error
		if (liso->feature[0] < m_featureBoundaries[0].minVal)
			m_featureBoundaries[0].minVal = liso->feature[0];
		if (liso->feature[0] > m_featureBoundaries[0].maxVal)
			m_featureBoundaries[0].maxVal = liso->feature[0];
		// grad
		if (liso->feature[1] < m_featureBoundaries[1].minVal)
			m_featureBoundaries[1].minVal = liso->feature[1];
		if (liso->feature[1] > m_featureBoundaries[1].maxVal)
			m_featureBoundaries[1].maxVal = liso->feature[1];
		// lambad
		if (liso->feature[2] < m_featureBoundaries[2].minVal)
			m_featureBoundaries[2].minVal = liso->feature[2];
		if (liso->feature[2] > m_featureBoundaries[2].maxVal)
			m_featureBoundaries[2].maxVal = liso->feature[2];
		// mass
		if (liso->feature[3] < m_featureBoundaries[3].minVal)
			m_featureBoundaries[3].minVal = liso->feature[3];
		if (liso->feature[3] > m_featureBoundaries[3].maxVal)
			m_featureBoundaries[3].maxVal = liso->feature[3];
	}
}


// add a set to training
void Trainer::addSet(std::vector<LISO>& lisoSet)
{
	int countLPIP = 0;
	int countNonLPIP = 0;
	for (int i = 0; i < (int)lisoSet.size(); i++)
	{
		LISO * liso = &lisoSet[i];
		double gammaCalibMin = (sqrt(3.)/(sqrt(3.)-1.)) * (1. - sqrt(1./(2.*(liso->gamma-0.1) + 1.)));
		double gammaCalibMax = (sqrt(3.)/(sqrt(3.)-1.)) * (1. - sqrt(1./(2.*(liso->gamma+0.1) + 1.)));

		if (liso->Q <= gammaCalibMax && liso->Q >= gammaCalibMin)
		{
			countLPIP++;
			m_lpip.push_back(*liso);
		}
		else
		{
			countNonLPIP++;
			m_nonlpip.push_back(*liso);
		}
	}
	std::cout << "    -> added " << countLPIP << " LPIPs and " << countNonLPIP << " NonLPIPs to training set" << std::endl;
}


// cimpute the histograms (normalized or not)
void Trainer::computeHists(bool normalized)
{
	for (int i = 0; i < NUM_FEATURES; i++)
	{
		m_featureHistLPIP[i] = new double[m_featureBoundaries[i].bins];
		m_featureHistNonLPIP[i] = new double[m_featureBoundaries[i].bins];
	}
	for (int i = 0; i < NUM_FEATURES; i++)
	{
		for (int j = 0; j < m_featureBoundaries[i].bins; j++)
		{
			m_featureHistLPIP[i][j] = 0;
			m_featureHistNonLPIP[i][j] = 0;
		}
	}

	// Do computations for LPIP set
	for (int i = 0; i < (int)m_lpip.size(); i++)
	{
		for (int j = 0; j < NUM_FEATURES; j++)
		{
			double value = m_lpip[i].feature[j];
			double min = m_featureBoundaries[j].minVal;
			double max = m_featureBoundaries[j].maxVal;
			double bins = m_featureBoundaries[j].bins;
			int pos = (int)((value-min)/(max-min)*(bins-1));
			m_featureHistLPIP[j][pos]++;
		}
	}

	// Do computations for nonLPIP set
	for (int i = 0; i < (int)m_nonlpip.size(); i++)
	{
		for (int j = 0; j < NUM_FEATURES; j++)
		{
			double value = m_nonlpip[i].feature[j];
			double min = m_featureBoundaries[j].minVal;
			double max = m_featureBoundaries[j].maxVal;
			double bins = m_featureBoundaries[j].bins;
			int pos = (int)((value-min)/(max-min)*(bins-1));
			m_featureHistNonLPIP[j][pos]++;
		}
	}

	if (normalized)
	{
		for (int i = 0; i < NUM_FEATURES; i++)
		{
			for (int j = 0; j < m_featureBoundaries[i].bins; j++)
			{
				m_featureHistLPIP[i][j] /= (double)m_lpip.size();
				m_featureHistNonLPIP[i][j] /= (double)m_nonlpip.size();
			}
		}
	}

	m_histComputed = true;
}


// prints the histograms to a file
void Trainer::printHists(std::string prefix)
{
	assert(m_histComputed);

	// create histogram files (one file per feature)
	// format: "binPosition  LPIP-Value  NonLPIP-Value"
	for (int i = 0; i < NUM_FEATURES; i++)
	{
		std::stringstream ss;
		ss << prefix << i << ".txt";
		std::ofstream file(ss.str().c_str());
		for (int j = 0; j < m_featureBoundaries[i].bins; j++)
		{
			double valueLPIP = m_featureHistLPIP[i][j];
			double valueNonLPIP = m_featureHistNonLPIP[i][j];
			double min = m_featureBoundaries[i].minVal;
			double max = m_featureBoundaries[i].maxVal;
			double bins = m_featureBoundaries[i].bins;
			double binPos = min + ((double)j/(double)bins*(max-min));
			file << binPos << " " << valueLPIP << " " << valueNonLPIP << std::endl;
		}
		file.close();
	}
}


// creates a file containing training results
// Format (for every feature 3 rows!):
// -------------------------------------------------
// P(LPIP) P(NonLPIP) <- only once!
// minVal maxVal numberOfBins
// <bin values (LPIP)> (numberOfBins bin values)
// <bin values (NonLPIP)> (numberOfBins bin values)
// "second feature..."
// ...
// -------------------------------------------------
void Trainer::saveResult(std::string filename)
{
	assert(m_histComputed);

	std::ofstream file(filename.c_str());

	file << (double)m_lpip.size() / ((double)m_lpip.size()+(double)m_nonlpip.size()) << " "
		 << (double)m_nonlpip.size() / ((double)m_lpip.size()+(double)m_nonlpip.size()) << std::endl;

	for (int i = 0; i < NUM_FEATURES; i++)
	{
		file << m_featureBoundaries[i].minVal << " "
			 << m_featureBoundaries[i].maxVal << " "
			 << m_featureBoundaries[i].bins << std::endl;
		for (int j = 0; j < m_featureBoundaries[i].bins; j++)
		{
			file << m_featureHistLPIP[i][j] << " ";
		}
		file << std::endl;
		for (int j = 0; j < m_featureBoundaries[i].bins; j++)
		{
			file << m_featureHistNonLPIP[i][j] << " ";
		}
		file << std::endl;
	}
	file.close();
}