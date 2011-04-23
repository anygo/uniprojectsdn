#include <iostream>
#include <fstream>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>

#include <ostream>  // ::std::ostream (<<) 
#include <ios>      // ::std::scientific, ::std::ios::scientific 
#include <iomanip>  // ::std::resetiosflags, ::std::setprecision


typedef struct INFO
{
	double imNum;
	double omSize;
	double cov;
	double blue;
	double green;
	double red;
} INFO;


int main (int argc, char *argv[])
{
	std::ofstream os("daten.txt");
	std::ofstream tables("tabellen.txt");
	for (int cameras = 0; cameras < 8; cameras++)
	{
		std::string cameraModel;
		std::string cameraName;
		switch (cameras)
		{
		case 0: cameraModel = "canon_eos_400d_digital"; cameraName = "Canon EOS 400D Rebel XTi"; break;
		case 1: cameraModel = "casio_ex-s600"; cameraName = "Casio Exilim EX-S600"; break;
		case 2: cameraModel = "fujifilm_finepix_s5600"; cameraName = "FujiFilm FinePix S5600"; break;
		case 3: cameraModel = "kodak_dx7590"; cameraName = "Kodak DX7590"; break;
		case 4: cameraModel = "kodak_z740"; cameraName = "Kodak Z740"; break;
		case 5: cameraModel = "nikon_d40"; cameraName = "Nikon D40"; break;
		case 6: cameraModel = "nikon_d80"; cameraName = "Nikon D80"; break;
		case 7: cameraModel = "sony_dsc-w300"; cameraName = "Sony CyberShot DSC-W300"; break;
		}
		for (int setting = 0; setting < 3; setting++)
		{
			int nKernels;
			int nPCAComponents;
			switch (setting)
			{
			case 0: nKernels = 3; nPCAComponents = 3; break;
			case 1: nKernels = 5; nPCAComponents = 5; break;
			case 2: nKernels = 7; nPCAComponents = 7; break;
			}

			for (double lambda = 100; lambda < 10001; lambda *= 10)
			{
				std::stringstream textfilename;
				textfilename << "stabilitytests/" << cameraModel << "_l" << lambda << "_k" << nKernels << "_p" << nPCAComponents << ".txt";
				std::ifstream f(textfilename.str().c_str());
					
				double B[3]; // mean min max
				double G[3];
				double R[3];
				double coverage[3];
				double omega[3];

				//initialize
				B[0] = G[0] = R[0] = coverage[0] = omega[0] = 0;
				B[1] = G[1] = R[1] = coverage[1] = omega[1] = 999999999;
				B[2] = G[2] = R[2] = coverage[2] = omega[2] = -999999999;
				
				std::vector<INFO> VEC_INFO;

				int count = 0;
				while (true)
				{
					INFO info;
					
					f >> info.imNum;
					f >> info.omSize;					
					f >> info.cov;					
					f >> info.blue;	info.blue /= 1024.;			
					f >> info.green; info.green /= 1024.;		
					f >> info.red; info.red /= 1024.;		

					if (f.eof())
						break;

					if (info.cov < 0.05)
						continue;

					VEC_INFO.push_back(info);

					B[0] += info.blue;
					G[0] += info.green;
					R[0] += info.red;
					coverage[0] += info.cov;
					omega[0] += info.omSize;

					B[1] = info.blue < B[1] ? info.blue : B[1];
					G[1] = info.green < G[1] ? info.green : G[1];
					R[1] = info.red < R[1] ? info.red : R[1];
					coverage[1] = info.cov < coverage[1] ? info.cov : coverage[1];
					omega[1] = info.omSize < omega[1] ? info.omSize : omega[1];

					B[2] = info.blue > B[2] ? info.blue : B[2];
					G[2] = info.green > G[2] ? info.green : G[2];
					R[2] = info.red > R[2] ? info.red : R[2];
					coverage[2] = info.cov > coverage[2] ? info.cov : coverage[2];
					omega[2] = info.omSize > omega[2] ? info.omSize : omega[2];

					count++;
				}
				f.close();

				double meanB = B[0]/(double)count;
				double meanG = G[0]/(double)count;
				double meanR = R[0]/(double)count;
				double meanAll = (B[0]+G[0]+R[0])/(double)(count*3);

				double varianzB = 0;
				double varianzG = 0;
				double varianzR = 0;
				double varianzAll = 0;
				for (int i = 0; i < (int)VEC_INFO.size(); i++)
				{
					INFO cur = VEC_INFO[i];
					varianzB += sqrt(pow((cur.blue - meanB), 2.));
					varianzG += sqrt(pow((cur.green- meanG), 2.));
					varianzR += sqrt(pow((cur.red - meanR), 2.));
					varianzAll += sqrt(pow((cur.blue - meanB), 2.)) + sqrt(pow((cur.green- meanG), 2.)) + sqrt(pow((cur.red - meanR), 2.));
				}
				varianzB /= (double)VEC_INFO.size();
				varianzG /= (double)VEC_INFO.size();
				varianzR /= (double)VEC_INFO.size();
				varianzAll /= (double)(VEC_INFO.size()*3);


				// write to file
				os << textfilename.str() << std::endl;
				os << "images: " << count << std::endl;
				os << "blue mean min max stddev" << std::endl;
				os << meanB << " " << R[1] << " " << R[2] << " " << varianzB << std::endl;
				os << "green mean min max stddev" << std::endl;
				os << meanG << " " << G[1] << " " << G[2] << " " << varianzG <<  std::endl;
				os << "red mean min max stddev" << std::endl;
				os << meanR << " " << B[1] << " " << B[2] << " " << varianzR <<  std::endl;
				os << "all curves mean stddev" << std::endl;
				os << meanAll << " " << varianzAll << std::endl;
				os << "coverage mean min max" << std::endl;
				os << coverage[0]/(double)count << " " << coverage[1] << " " << coverage[2] << std::endl;
				os << "omega mean min max" << std::endl;
				os << omega[0]/(double)count << " " << omega[1] << " " << omega[2] << std::endl;
				os << std::endl;

				tables << ::std::fixed << ::std::setw(2);// << std::endl;
				tables << "\\begin{table}" << std::endl
					   << "  \\centering" << std::endl
					   << "    \\begin{tabular}{|c||c|c|c|c|}\\hline" << std::endl
					   << "		   & \\textbf{Red} & \\textbf{Green} & \\textbf{Blue} & \\textbf{$\\sum$} \\\\\\hline\\hline" << std::endl
					   << "      $e_\\text{min}$ & " << R[1]     << " & " << G[1]     << " & " << B[1]     << " & " << " "        << " \\\\\\hline" << std::endl
					   << "      $e_\\text{max}$ & " << R[2]     << " & " << G[2]     << " & " << B[2]     << " & " << " "        << " \\\\\\hline" << std::endl
					   << "      $\\mu$          & " << meanR    << " & " << meanG    << " & " << meanB    << " & \\textbf{" << meanAll    << "} \\\\\\hline" << std::endl
					   << "      $\\sigma$       & " << varianzR << " & " << varianzG << " & " << varianzB << " & " << varianzAll << " \\\\\\hline" << std::endl
					   << "    \\end{tabular}" << std::endl
					   << "  \\caption{" << cameraName << " -- $\\kappa$ = " << nKernels << "; $\\nu$ = " << nPCAComponents << "; $\\lambda$ = " << (int)lambda << "}" << std::endl
					   << "  \\label{tab:" << cameraName << nKernels << nPCAComponents << (int)lambda << "}" << std::endl
					   << "\\end{table}" << std::endl;
				
			}
		}
		tables << std::endl << "\\clearpage" << std::endl;
	}

	os.close();
	tables.close();
}