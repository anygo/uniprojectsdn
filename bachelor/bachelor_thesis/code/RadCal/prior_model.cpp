#include "prior_model.h"
#include <fstream>
#include <iomanip>


using namespace cv;

PriorModel::PriorModel(const char *filenameInverseResponseFunctionsPCA, int nKernels, int nPCAComponents)
{
	m_nKernels = nKernels;
	m_nPCAComponents = nPCAComponents;
	m_samples = readMatrix(filenameInverseResponseFunctionsPCA).colRange(0, m_nPCAComponents);
	
	setEMParams();
	m_EMmodel.train(m_samples, Mat(), m_EMparams, 0);


	/*std::cout << "Priors (Equation 9): " << std::endl;
	for (int i = 0; i < 201; i++)
	{
		Mat1f cur = m_samples.row(i);
		std::cout << i << ":\t" << std::fixed << prior(cur) << ",\t-log: " << -std::log(prior(cur)) << std::endl;
	}
	std::cout << "^ 201 priors from eq 9" << std::endl;

	Mat1f meanCurveComponents = Mat1f::zeros(1, m_nPCAComponents);
	std::cout << "prior of mean curve: " << prior(meanCurveComponents) << std::endl << std::endl;*/
}


double PriorModel::prior(cv::Mat1f &coefficients)
{
	Mat1f means = Mat1f(m_EMmodel.get_means());
	Mat1f weights = Mat1f(m_EMmodel.get_weights());

	// only one coefficient vector at a time
	assert(coefficients.rows == 1 && coefficients.cols == m_nPCAComponents);

	double result = 0;
	for (int i = 0; i < m_nKernels; i++)
	{
		Mat1f cov = Mat1f(m_EMmodel.get_covs()[i]);
		Mat1f mean = means.row(i);

		double weight = weights[0][i];
		double nd = NormalDistribution(cov, mean, coefficients);
		result += weight * nd;
	}
	return result;
}

//double PriorModel::prior(int index)
//{
//	Mat1f weights = m_EMmodel.get_weights();
//	Mat1f probs = m_EMmodel.get_probs();
//	probs = probs.row(index);
//
//	// FIX/DELETE ME
//	Mat1f prob;
//	Mat1f test = Mat1f::ones(1,5)*(0);
//
//	m_EMmodel.predict(test, &prob);
//	probs = prob.t();
//
//	double sum = 0;
//	for (int i = 0; i < m_nClusters; i++)
//	{
//		double weight = weights[0][i];
//		double prob = probs[0][i];
//		sum += weight * prob; // OK??? is prob == "N"?
//	}
//	return sum;
//}


void PriorModel::setEMParams()
{
	m_EMparams.covs      = NULL; 
    m_EMparams.means     = NULL; 
    m_EMparams.weights   = NULL; 
    m_EMparams.probs     = NULL; 
    m_EMparams.nclusters = m_nKernels; 
	m_EMparams.cov_mat_type       = CvEM::COV_MAT_DIAGONAL;
    m_EMparams.start_step         = CvEM::START_AUTO_STEP; 
    m_EMparams.term_crit.max_iter = 10; 
    m_EMparams.term_crit.epsilon  = 0.001; 
    m_EMparams.term_crit.type     = CV_TERMCRIT_ITER|CV_TERMCRIT_EPS;
}

cv::Mat1f PriorModel::readMatrix(const char *filename)
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

	//normalize(m,m,1,0,NORM_MINMAX);
	// FIXME
	/*for (int i = 0; i < m.cols; i++)
	{
		cv::normalize(m.col(i), m.col(i), 1, -1, NORM_MINMAX);
	}
	cv::threshold(m, m, 0.00001, 0, CV_THRESH_TOZERO);
	std::cout << "Achtung, pseudo normalisierte werte ;-)" << std::endl;*/
	////m = m*100;
	
	/*std::ofstream file("normalized.txt");
	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			file << m[i][j] << " ";
		}
		file << std::endl;
	}
	file.close();*/

	return m;
}


// computes normal distribution for a N-dimensional vector x using a
// NxN diagonal covariance matrix and a N-Dimensional mean vector
double PriorModel::NormalDistribution(cv::Mat1f &cov, cv::Mat1f mean, cv::Mat1f x)
{
		
	/*std::cout << "cov:\t";
	for (int i = 0; i < cov.rows; i++)
	{
		std::cout << cov[i][i] << " ";
	}
	std::cout << std::endl;*/

	if (mean.cols > mean.rows) 
		mean = mean.t();
	if (x.cols > x.rows) 
		x = x.t();

	
	Mat1f expTerm = x - mean;
	expTerm = expTerm.t();
	expTerm *= cov.inv();
	expTerm *= (x - mean);
	expTerm *= -0.5;
	double bla = expTerm[0][0];	
	double leftSide = 1. / ( pow(2*CV_PI, cov.rows/2.) * sqrt(determinant(cov)));
	double result = leftSide * exp(expTerm[0][0]);


	// Christians stuff (same result)
	/*double expStuff = 0;
	double prod = 1;
	for (int i = 0; i < cov.rows; i++) {
		expStuff += (x[0][i] - mean[0][i]) * (x[0][i] - mean[0][i]) * 1.0/ cov[i][i];
		prod *= cov[i][i];
	}
	double otherResult = (1.0 / (pow(2 * CV_PI, static_cast<float>(cov.rows) / 2.0) * sqrt(prod)))
		* exp(-0.5 * expStuff);
	if (otherResult > 1) std::cout << "neues Ergebnis: " << otherResult << std::endl;
	return otherResult;*/


	//double ret = 1;
	//for (int i = 0; i < cov.rows; i++)
	//{
	//	double expTerm = -pow((x[0][i] - mean[0][i]), 2) / (2*cov[i][i]*cov[i][i]);
	//	double leftSide = 1./sqrt(2*CV_PI*cov[i][i]*cov[i][i]);
	//	double result = leftSide * exp(expTerm);
	//	//std::cout << ", " << result;
	//	ret *= result;
	//}
	//std::cout << std::endl << "ret: " << ret << std::endl;

	/*if (result > 1.) 
	{
		
		std::cout << "cov:\t";
		for (int i = 0; i < cov.rows; i++)
		{
			std::cout << cov[i][i] << " ";
		}
		std::cout << std::endl;

		std::cout << "mean:\t";
		for (int i = 0; i < mean.rows; i++)
		{
			std::cout << mean[i][0] << " ";
		}
		std::cout << std::endl;

		std::cout << "x:\t";
		for (int i = 0; i < x.rows; i++)
		{
			std::cout << x[i][0] << " ";
		}
		std::cout << std::endl;
	}*/
	//
	//return ret;

	return result;
}
