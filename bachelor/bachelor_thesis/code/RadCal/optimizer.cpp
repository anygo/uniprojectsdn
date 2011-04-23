#include "optimizer.h"
#include "lmcurve.h"
#include <fstream>


using namespace cv;

// constructor
Optimizer::Optimizer(PriorModel &pm, LikelihoodFunction &lf, const std::vector<ColorTriple> &omega)
{
	m_pm = &pm;
	m_lf = &lf;
	m_omega = &omega;
}

// objective function from the paper
double Optimizer::objectiveFuntion(cv::Mat1f &coefficients)
{
	double lambda = m_lf->getLambda();
	double distance = m_lf->getTotalDistance(*m_omega, coefficients);
	
	// merged prior (3 channels!)
	double prior = 1; 
	for (int i = 0; i < coefficients.rows; i++)
	{
		cv::Mat1f tmp = coefficients.row(i);
		prior *= m_pm->prior(tmp);
	}
	prior = std::pow(prior, 1./(double)coefficients.rows);
	
	double left = lambda * distance;
	double right = std::log(prior < DBL_MIN ? DBL_MIN : prior);
	double result = left - right;

	return result;
}


// used by LM routine
void evalLM(const double *par, int m_dat, const void *data, double *fvec, int *info)
{
	Optimizer *opt = (Optimizer *)data;

	cv::Mat1f coeffs(3, m_dat/3); // 3 channels!!
	for (int i = 0; i < m_dat/3; i++)
	{
		int offset = opt->m_pm->getNPCAComponents();
		coeffs[0][i] = static_cast<float>(par[i+0*offset]);
		coeffs[1][i] = static_cast<float>(par[i+1*offset]);
		coeffs[2][i] = static_cast<float>(par[i+2*offset]);
	}
	
	fvec[0] = opt->objectiveFuntion(coeffs);
	for (int i = 1; i < m_dat; i++)
	{
		fvec[i] = fvec[0];
	}
}

// levenberg-marquardt, verbosity...
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


// starts the optimization
std::string Optimizer::optimize(int maxIter)
{
	int n_par = m_pm->getNPCAComponents();
	n_par *= 3; // 3 channels!!
    double *par = new double[n_par];
	for (int i = 0; i < n_par; i++)
	{
		par[i] = 0;
	}

    int m_dat = n_par;

    lm_status_struct status;
	lm_control_struct control = lm_control_float;
    control.printflags = 3; 
	control.maxcall = maxIter;

	std::cout << "LM Optimization..." << std::endl;
	lmmin(n_par, par, m_dat, (void*)this, evalLM, &control, &status, lm_printout_mine);

    /* print results */
	std::cout << std::endl << "Status after " << status.nfev << " function evaluations:" << std::endl;
	std::cout << "  -> " << lm_infmsg[status.info] << std::endl;


	// best coefficients after Levenberg Marquardt
	m_bestCoefficients = Mat1f(3, m_pm->getNPCAComponents());
	for (int i = 0; i < m_pm->getNPCAComponents(); i++)
	{
		int offset = m_pm->getNPCAComponents();
		m_bestCoefficients[0][i] = static_cast<float>(par[i+0*offset]);
		m_bestCoefficients[1][i] = static_cast<float>(par[i+1*offset]);
		m_bestCoefficients[2][i] = static_cast<float>(par[i+2*offset]);
	}

	
	//std::cout << std::endl << "Refining in each dimension..." << std::endl;

	double *parRefined = new double[n_par];
	for (int i = 0; i < n_par; i++)
	{
		parRefined[i] = par[i];
	}
	
	// refining in each dimension
	double bestResult = objectiveFuntion(m_bestCoefficients);
	int refinements = 0;
	for (int params = 0; params < n_par; params++)
	{
		for (int j = 0; j < 2; j++)
		{
			double change;
			if (j == 0) change = control.epsilon*10;
			else change = -control.epsilon*10;
			while (true)
			{
				parRefined[params] += change;
				cv::Mat1f coeffs(3, m_dat/3); // 3 channels!!
				for (int i = 0; i < m_dat/3; i++)
				{
					int offset = m_pm->getNPCAComponents();
					coeffs[0][i] = static_cast<float>(parRefined[i+0*offset]);
					coeffs[1][i] = static_cast<float>(parRefined[i+1*offset]);
					coeffs[2][i] = static_cast<float>(parRefined[i+2*offset]);
				}
				
				double result = objectiveFuntion(coeffs);

				if (result <= bestResult)
				{
					refinements++;
					bestResult = result;
				}
				else
				{
					parRefined[params] -= change;
					break;
				}
				if (refinements > 25)
					break;
			}
		}
	}

	// initialize "best" member variables
	m_bestCoefficients = Mat1f(3, m_pm->getNPCAComponents());
	for (int i = 0; i < m_pm->getNPCAComponents(); i++)
	{
		int offset = m_pm->getNPCAComponents();
		m_bestCoefficients[0][i] = static_cast<float>(par[i+0*offset]);
		m_bestCoefficients[1][i] = static_cast<float>(par[i+1*offset]);
		m_bestCoefficients[2][i] = static_cast<float>(par[i+2*offset]);
	}
	m_bestFunction = m_lf->computeFunctions(m_bestCoefficients);

	free(par);
	free(parRefined);

	return std::string(lm_infmsg[status.info]);
}


// one function per column (blue green red channels) -> estimated CRF
void Optimizer::printBestFunction(const char *filename)
{
	std::ofstream file(filename);
	for (int i = 0; i < m_bestFunction.cols; i++)
	{
		file << static_cast<float>((i+1.)/(double)m_bestFunction.cols) << " "; // first axis (normalized)
		for (int j = 0; j < m_bestFunction.rows; j++)
		{
			file << m_bestFunction[j][i] << " ";
		}
		file << std::endl;
	}
	std::cout << std::endl;
	file.close();	
}


// one function per column (blue green red channels) - for any function
void Optimizer::printFunction(const char *filename, cv::Mat1f function)
{
	std::ofstream file(filename);
	for (int i = 0; i < function.cols; i++)
	{
		file << static_cast<float>((i+1.)/(double)function.cols) << " "; // first axis (normalized)
		for (int j = 0; j < function.rows; j++)
		{
			file << function[j][i] << " ";
		}
		file << std::endl;
	}
	std::cout << std::endl;
	file.close();	
}


// plots the best function in an opencv namedWindow (not really used)
void Optimizer::plotFunction(cv::Mat1f function)
{
	Mat3b plot = Mat3b(512, 512);
	plot.setTo(Scalar::all(255));

	for (int i = 0; i < function.rows; i++)
	{
		for (int j = 0; j < 512-1; j++)
		{
			Point from(j, 511-min(511, (int)(function[i][j*2]*512)));
			Point to(j+1, 511-min(511, (int)(function[i][(j+1)*2]*512)));
			line(plot, from, to, i == 0 ? Scalar(255,0,0) : i == 1 ? Scalar(0,255,0) : Scalar(0,0,255), 1);
		}
	}
	
	namedWindow("plot", 1);
	imshow("plot", plot);
	waitKey(5);
}


// applies the estimated function on an image
void Optimizer::applyBestFunction(cv::Mat3b &image)
{
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			image[i][j].val[0] = m_bestFunction[0][(int)(((double)image[i][j].val[0]/255.)*(double)m_bestFunction.cols)]*255;
			image[i][j].val[1] = m_bestFunction[1][(int)(((double)image[i][j].val[1]/255.)*(double)m_bestFunction.cols)]*255;
			image[i][j].val[2] = m_bestFunction[2][(int)(((double)image[i][j].val[2]/255.)*(double)m_bestFunction.cols)]*255;
		}
	}
}
