#include "classifier.h"
#include <fstream>


using namespace cv;


// constructor, only the image is loaded
Classifier::Classifier(std::string filename)
{
	load(filename);
}


// read training results
void Classifier::load(std::string filename)
{
	std::ifstream file(filename.c_str());
	file >> m_PLPIP;
	file >> m_PNonLPIP;
	
	for (int i = 0; i < NUM_FEATURES; i++)
	{
		file >> m_featureBoundaries[i].minVal;
		file >> m_featureBoundaries[i].maxVal;
		file >> m_featureBoundaries[i].bins;
		m_featureHistLPIP[i] = new double[m_featureBoundaries[i].bins];
		for (int j = 0; j < m_featureBoundaries[i].bins; j++)
		{
			file >> m_featureHistLPIP[i][j];
		}
		m_featureHistNonLPIP[i] = new double[m_featureBoundaries[i].bins];
		for (int j = 0; j < m_featureBoundaries[i].bins; j++)
		{
			file >> m_featureHistNonLPIP[i][j];
		}
	}
}


// the probability function
double Classifier::probabilityFunction(double featureVector[NUM_FEATURES])
{
	// positions in "histograms" for the given feature vector
	int histPositions[NUM_FEATURES];
	for (int i = 0; i < NUM_FEATURES; i++)
	{
		double value = featureVector[i];
		double minVal = m_featureBoundaries[i].minVal;
		double maxVal = m_featureBoundaries[i].maxVal;
		int bins = m_featureBoundaries[i].bins;

		histPositions[i] = min(bins-1, max(0, (int)( ((value-minVal)/(maxVal-minVal))*(bins-1) )));
	}

	// P(f|LPIP)
	double PfLPIP = 1.;
	for (int i = 0; i < NUM_FEATURES; i++)
	{
		double tmp = m_featureHistLPIP[i][histPositions[i]] / 
			(m_featureHistLPIP[i][histPositions[i]] + m_featureHistNonLPIP[i][histPositions[i]]);
		PfLPIP *= tmp;
	}

	// P(f|nonLPIP)
	double PfNonLPIP = 1.;
	for (int i = 0; i < NUM_FEATURES; i++)
	{
		double tmp = m_featureHistNonLPIP[i][histPositions[i]] / 
			(m_featureHistLPIP[i][histPositions[i]] + m_featureHistNonLPIP[i][histPositions[i]]);
		PfNonLPIP *= tmp;
	}


	double numerator = PfLPIP*m_PLPIP;
	double denominator = PfLPIP*m_PLPIP + PfNonLPIP*m_PNonLPIP;
	double result = numerator / denominator;

	return result;
}


// creates RQ histogram and saves it as an image
void Classifier::createHistRQ(int binsX, int binsY, std::vector<LISO>& lisoSet, bool weighted, std::string title)
{
	Mat1f bins = Mat1f::zeros(binsY, binsX);

	for (int i = 0; i < (int)lisoSet.size(); i++)
	{
		LISO cur = lisoSet[i];
		if (cur.Q < 0.0 || cur.Q > 1.5)
			continue;

		int posX = (int)(cur.R*(double)(binsX-1));
		int posY = (int)((binsY-1) - (cur.Q/1.5)*(double)(binsY-1));

		if (posX >= binsX || posY >= binsY || posX < 0 || posY < 0)
			continue;
		double toAdd = 1.;
		if (weighted)
		{
			double weight = probabilityFunction(cur.feature);
			toAdd *= weight;
		}
		bins[posY][posX] += (float)toAdd;
	}

	normalize(bins, bins, 1, 0, NORM_MINMAX);

	// new
	std::stringstream titlette;
	titlette << title << ".txt";
	std::ofstream histi(titlette.str().c_str());
	for (int i = 0; i < bins.rows; i++)
		for (int j = 0; j < bins.cols; j++)
		{
			histi << 1.5-i/double(bins.rows-1)*1.5 << " " << j/double(bins.cols-1) << " " << bins[i][j] << std::endl;
		}
	histi.close();
	// end new

	Mat1f tmp;
	resize(bins, tmp, Size(600, 180), 0, 0, INTER_NEAREST);	
	bins = tmp;

	std::stringstream ss;
	ss << title << ".png";
	imwrite(ss.str(), bins*255);
}