#include "crf_estimator.h"
#include <iostream>
#include "lmcurve.h"
#include <math.h>
#include <fstream>


using namespace cv;


CRFEstimator::CRFEstimator(Classifier& classifier, std::vector<LISO>& lisoSet)
{
	m_classifier = &classifier;
	m_lisoSet = lisoSet;
	
	// "Caching" for P(R_k), assuming R to be a floating point value between 0..1
	for (int i = 0; i < BINS_FOR_R; i++)
	{
		m_distributionR[i] = 0.;
	}

	for (int i = 0; i < (int)m_lisoSet.size(); i++)
	{
		int pos = (int)(m_lisoSet[i].R * ((double)BINS_FOR_R-1.));
		assert(pos >= 0 && pos < BINS_FOR_R);
		m_distributionR[pos] += 1.;
	}

	for (int i = 0; i < BINS_FOR_R; i++)
	{
		m_distributionR[i] /= (double)m_lisoSet.size();
	}
}

double CRFEstimator::functionQ(double R, double alpha0, double alpha1, bool calibrated)
{
	assert(R > DBL_EPSILON);
	double T = pow(alpha0, 2) + alpha0*alpha1*R * (alpha0* (std::log(R) + 1.) - 2.* (1. - std::log(R))) +
		pow(alpha1*R, 2) * (1. - 4.*alpha0 - 2.*alpha1*R + (std::log(R) - 2.)*(alpha1*R + std::log(R)));
	double top = pow((alpha0 + alpha1*R), 2) * (alpha1 * std::log(R) - alpha0 + alpha1*R);

	double Q = top / T;



	if (!calibrated)
		return abs(Q);
	else
		return (sqrt(3.)/(sqrt(3.)-1.)) * (1. - sqrt(1./(2.*abs(Q) + 1.)));
}

double CRFEstimator::objectiveFunction(double alpha0, double alpha1)
{
	double res = 0;
	
	for (int i = 0; i < (int)m_lisoSet.size(); i++)
	{
		double R = m_lisoSet[i].R;
		double Q = m_lisoSet[i].Q;

		// computing P(Q_j|R_k)
		// computing P(R_k)
		int pos = (int)(R*((double)BINS_FOR_R-1.));
		assert(pos <= BINS_FOR_R-1);
		double PRk = m_distributionR[pos]; // precomputed in constructor!


		// computing P(Q_j, R_k)
		double PQjRk = m_classifier->probabilityFunction(m_lisoSet[i].feature);


		double left = PQjRk / PRk;		
		double right = Q - functionQ(R, alpha0, alpha1, true);
		right *= right;

		double toAdd = left * right;

		res += toAdd;
	}

	return res;
}


void evalLM(const double *par, int m_dat, const void *data, double *fvec, int *info)
{
	CRFEstimator *estimator = (CRFEstimator *)data;
	
	fvec[0] = estimator->objectiveFunction(par[0], par[1]);
	for (int i = 1; i < m_dat; i++)
	{
		fvec[i] = fvec[0];
	}

	estimator->plot(par[0], par[1]);
	if (waitKey(5) > 0) *info = -1;
}

void lm_printout_mine( int n_par, const double *par, int m_dat,
                      const void *data, const double *fvec,
                      int printflags, int iflag, int iter, int nfev)
{
    if( !printflags )
        return;


    if( printflags & 1 )
	{
        if (iflag == -1) 
		{
			std::cout << "\rterminated after " << nfev << " evaluations";
        }
		else
		{
			std::cout << "\riteration " << iter << ":\t";
		}
    }

    if( printflags & 2 )
	{
		std::cout << " norm: " << lm_enorm(m_dat, fvec);
    }
}

void CRFEstimator::estimate()
{
	int n_par = 2;
    double *par = new double[n_par];
	// most CRFs lie in between gamma = 0.2..0.6 -> 0.4 seems to be a good guess
	par[0] = 0.4;
	par[1] = 0.0;

    int m_dat = n_par;

    lm_status_struct status;
	lm_control_struct control = lm_control_double;
    control.printflags = 3; 
	control.maxcall = 150;

	std::cout << "LM Optimization..." << std::endl;
	lmmin(n_par, par, m_dat, (void*)this, evalLM, &control, &status, lm_printout_mine);

    /* print results */
	std::cout << std::endl << "Results:" << std::endl;
	std::cout << "Status after " << status.nfev << " function evaluations:" << std::endl;
	std::cout << "  -> " << lm_infmsg[status.info] << std::endl;

	std::cout << "alpha0: " << par[0] << std::endl;
	std::cout << "alpha1: " << par[1] << std::endl;
}


void CRFEstimator::plot(double alpha0, double alpha1)
{
	int size = 600;
	Mat3f img(size, size);

	img.setTo(Scalar::all(1));

	for (int i = 0; i < size; i++)
	{
		double x = (double)i/(double)size;
		double val1 = pow(x, alpha0 + alpha1*x);
		double val2 = pow(x, 1./(alpha0 + alpha1*x));
		if (val1 <= 1 && val1 >= 0) 
		{
			int y = (int)(val1*((double)size-1.));
			img[(size-1)-y][i] = Vec3f(1,0,0);
		}

		if (val2 <= 1 && val2 >= 0)
		{
			int y = (int)(val2*((double)size-1.));
			img[(size-1)-y][i] = Vec3f(0,0,1);
		}
	}

	imshow("plot", img);
}