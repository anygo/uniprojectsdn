
#include "observation_set_formator.h"


using namespace cv;

// after calling the constructor, call this function compute
// the formation set
void ObservationSetFormator::formateObservationSet()
{
	createEdgeImage(m_image, m_edges);
	Mat3f DELETEME;

	for (int i = 0; i < m_image.rows - m_patchsize; i += m_stepWidthY)
	{
		int j = 0;
		while (j < m_image.cols - m_patchsize)
		{
			Rect r = Rect(j, i, m_patchsize, m_patchsize);

			int nonZeroElements = countNonZero(m_edges(r));
			if (nonZeroElements < m_patchsize || nonZeroElements > 1.5 * m_patchsize)
			{
				j += m_stepWidthX;
			}
			else if (checkPatch(r))
			{
				j += m_patchsize;

				// paint rectangle (patches will not overlap any more)
				rectangle(m_edges, r, Scalar(UCHAR_MAX/2), 1);
			}
			else
			{
				j += m_stepWidthX;
			}
		}
	}
}


// checks a particular ColorTriple (meeting several conditions:
// colorDifferenceThreshold and variance) and - as appropriate - adds
// the given ColorTriple to the ObservationSet
bool ObservationSetFormator::checkAndAddColorTriple(ColorTriple &colorTriple)
{
	if (colorTriple.distM1 > m_maxDistanceInRegion ||
		colorTriple.distM2 > m_maxDistanceInRegion)
		return false;

	// Euclidean norm
	cv::Vec3f M1M2 = colorTriple.M1 - colorTriple.M2;
	double distM1M2 = sqrt(M1M2.val[0]*M1M2.val[0] + 
		M1M2.val[1]*M1M2.val[1] + 
		M1M2.val[2]*M1M2.val[2]);
	if (distM1M2 < m_minDistanceBetweenRegions)
		return false;

	// exclude saturated pixels
	for (int i = 0; i < 3; i++)
	{
		if (colorTriple.M1[i] > 1.-FLT_EPSILON ||
			colorTriple.M2[i] > 1.-FLT_EPSILON ||
			colorTriple.Mp[i] > 1.-FLT_EPSILON)
			//std::cout << "saturation reached" << std::endl;
		return false;
	}	

	m_observationSet.push_back(colorTriple);
	return true;
}


// helper function: creates a (Canny) edge image out of an BGR-Image
void ObservationSetFormator::createEdgeImage(cv::Mat3f &src, cv::Mat1f &dst)
{
	cvtColor(src, dst, CV_BGR2GRAY);
	GaussianBlur(dst, dst, cv::Size(7,7), 0.5, 0.5);
	Mat1b tmp = dst*255;
	Canny(tmp, tmp, m_cannyThreshold1, m_cannyThreshold2, m_cannyApertureSize, true);
	dst = tmp*(1./255.);
}


// checks, if a particular patch (given by a rectangular) complies
// and - as appropriate - "generates" a ColorTriple out of it and gets that
// ColorTriple checked and added to the ObservationSet by checkAndAddColorTriple()
bool ObservationSetFormator::checkPatch(const cv::Rect &r)
{
	Mat1f edgePatchOrig = m_edges(r);
	Point start = scanBorder(edgePatchOrig);

	if (start.x == -1) // no edge pixel found in first row/column
		return false;

	
	Mat1f edgePatchTmp = edgePatchOrig.clone();
	Mat3f imagePatchOrig = m_image(r);


	std::vector<PIXEL_INFO> vector;
	getMeanColor(vector, edgePatchTmp, imagePatchOrig, start, true);


	if (vector.size() == 0) 
		return false;
	Point end = vector.at(0).coords;

	if (end.x != 0 && end.x != m_patchsize-1 && end.y != 0 && end.y != m_patchsize-1)
		return false;


	// line should pass midpoint
	bool lineOK = false;
	for (unsigned int i = 0; i < vector.size(); i++)
	{
		Point dist = vector.at(i).coords - Point(m_patchsize/2, m_patchsize/2);
		if (abs(dist.x) + abs(dist.y) < std::max(m_patchsize/6, 1))
		{
			lineOK = true;
			break;
		}
	}

	if (!lineOK)
		return false;

	dilate(edgePatchTmp, edgePatchTmp, m_dilateKernel);
	
	ColorTriple ct;
	ct.rect = Rect(r);


	ct.Mp = Vec3f::all(0);
	ct.Mp += vector.at(rand()%(int)vector.size()).color;


	ct.distMp = DBL_MIN;
	for (int i = 0; i < (int)vector.size(); i++)
	{
		for (int j = 1; j < (int)vector.size(); j++)
		{
			Vec3f sub = vector.at(i).color - vector.at(j).color;
			double norm = sqrt(sub.val[0]*sub.val[0] + sub.val[1]*sub.val[1] + sub.val[2]*sub.val[2]);
			ct.distMp = norm > ct.distMp ? norm : ct.distMp;
		}
	}


	// get M1
	vector.clear();
	start.x = 0;
	start.y = 0;
	getMeanColor(vector, edgePatchTmp, imagePatchOrig, start, false);
	if (vector.size() == 0) return false;

	ct.M1 = Vec3f::all(0);
	for (int i = 0; i < (int)vector.size(); i++)
	{
		ct.M1 += vector.at(i).color;
	}
	for (int i = 0; i < 3; i++)
	{
		ct.M1.val[i] /= (float)vector.size();
	}

	ct.distM1 = DBL_MIN;
	for (int i = 0; i < (int)vector.size(); i++)
	{
		for (int j = 1; j < (int)vector.size(); j++)
		{
			Vec3f sub = vector.at(i).color - vector.at(j).color;
			double norm = sqrt(sub.val[0]*sub.val[0] + sub.val[1]*sub.val[1] + sub.val[2]*sub.val[2]);
			ct.distM1 = norm > ct.distM1 ? norm : ct.distM1;
		}
	}


	// get M2
	vector.clear();
	start.x = m_patchsize-1;
	start.y = m_patchsize-1;
	getMeanColor(vector, edgePatchTmp, imagePatchOrig, start, false);
	if (vector.size() == 0) return false;
	
	ct.M2 = Vec3f::all(0);
	for (int i = 0; i < (int)vector.size(); i++)
	{
		ct.M2 += vector.at(i).color;
	}
	for (int i = 0; i < 3; i++)
	{
		ct.M2.val[i] /= (float)vector.size();
	}

	ct.distM2 = DBL_MIN;
	for (int i = 0; i < (int)vector.size(); i++)
	{
		for (int j = 1; j < (int)vector.size(); j++)
		{
			Vec3f sub = vector.at(i).color - vector.at(j).color;
			double norm = sqrt(sub.val[0]*sub.val[0] + sub.val[1]*sub.val[1] + sub.val[2]*sub.val[2]);
			ct.distM2 = norm > ct.distM2 ? norm : ct.distM2;
		}
	}

	// generate five color triples from one edge patch (4 remain)
	ColorTriple ct2;
	ColorTriple ct3;
	ColorTriple ct4;
	ColorTriple ct5;

	ct2.distM1 = ct3.distM1 = ct4.distM1 = ct5.distM1 = ct.distM1;
	ct2.distM2 = ct3.distM2 = ct4.distM2 = ct5.distM2 = ct.distM2;
	ct2.distMp = ct3.distMp = ct4.distMp = ct5.distMp = ct.distMp;
	ct2.M1 = ct3.M1 = ct4.M1 = ct5.M1 = ct.M1;
	ct2.M2 = ct3.distM2 = ct4.distM2 = ct5.distM2 = ct.distM2;

	ct2.Mp += vector.at(rand()%(int)vector.size()).color;
	ct3.Mp += vector.at(rand()%(int)vector.size()).color;
	ct4.Mp += vector.at(rand()%(int)vector.size()).color;
	ct5.Mp += vector.at(rand()%(int)vector.size()).color;

	checkAndAddColorTriple(ct2);
	checkAndAddColorTriple(ct3);
	checkAndAddColorTriple(ct4);
	checkAndAddColorTriple(ct5);
	return checkAndAddColorTriple(ct);
}


// recursive method for getting the mean color of a specific
// patch of the image (bounded by the patch borders AND the edge
// image of the patch) -> some kind of an implementation of the well 
// known seed-fill algorithm
// the parameter white (boolean) can be used as a switch, so that 
// this function can also be used for determing the mean color 
// along the edge path.
void ObservationSetFormator::getMeanColor(std::vector<PIXEL_INFO> &vector, 
										  cv::Mat1f &edgePatch, cv::Mat3f &imagePatch, 
										  cv::Point start, 
										  bool white
										  )
{
	// break conditions
	if (start.x < 0 || start.x >= m_patchsize) return;
	if (start.y < 0 || start.y >= m_patchsize) return;
	if (white)
	{
		if (edgePatch[start.y][start.x] < 0.9) return;
	}
	else
	{
		if (edgePatch[start.y][start.x] > 0) return;
	}

	// mark pixel in edge patch
	edgePatch[start.y][start.x] = 0.5;
	

	getMeanColor(vector, edgePatch, imagePatch, Point(start.x, start.y + 1), white);
	getMeanColor(vector, edgePatch, imagePatch, Point(start.x, start.y - 1), white);
	getMeanColor(vector, edgePatch, imagePatch, Point(start.x + 1, start.y + 1), white);
	getMeanColor(vector, edgePatch, imagePatch, Point(start.x + 1, start.y - 1), white);
	getMeanColor(vector, edgePatch, imagePatch, Point(start.x + 1, start.y), white);
	getMeanColor(vector, edgePatch, imagePatch, Point(start.x - 1, start.y), white);
	getMeanColor(vector, edgePatch, imagePatch, Point(start.x - 1, start.y + 1), white);
	getMeanColor(vector, edgePatch, imagePatch, Point(start.x - 1, start.y - 1), white);

	PIXEL_INFO pinf;
	pinf.coords.x = start.x;
	pinf.coords.y = start.y;
	pinf.color = imagePatch[start.y][start.x];

	vector.push_back(pinf);
}


// scan first column/row and check, whether there is any edge-pixel
// If found, return the coordinates of that pixel in relation to the
// given edge patch
cv::Point ObservationSetFormator::scanBorder(cv::Mat1f &edgePatch)
{
	Point start(-1, -1);
	for (int i = 0; i < m_patchsize; i++)
	{
		if (edgePatch[i][0] > 0)
		{
			start.x = 0;
			start.y = i;
			break;
		}
		else if (edgePatch[0][i] > 0)
		{
			start.x = i;
			start.y = 0;
			break;
		}
	}

	return start;
}

// returns color coverage factor (beta) for 16*16*16 clusters
double ObservationSetFormator::getCoverage()
{
	cv::Mat3b img = m_image*256;
	img *= 1./16.;

	int clusters[17][17][17];
	for (int i = 0; i < 17; i++)
		for (int j = 0; j < 17; j++)
			for (int k = 0; k < 17; k++)
				clusters[i][j][k] = 0;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			clusters[img[i][j].val[0]][img[i][j].val[1]][img[i][j].val[2]] = 1;
		}
	}

	for (int i = 0; i < (int)m_observationSet.size(); i++)
	{
		int a = (int)(m_observationSet.at(i).M1.val[0]*16);
		int b = (int)(m_observationSet.at(i).M1.val[1]*16);
		int c = (int)(m_observationSet.at(i).M1.val[2]*16);
		if (clusters[a][b][c] >= 1)
			clusters[(int)(m_observationSet.at(i).M1.val[0]*16)][(int)(m_observationSet.at(i).M1.val[1]*16)][(int)(m_observationSet.at(i).M1.val[2]*16)]++;
		if (clusters[(int)(m_observationSet.at(i).M2.val[0]*16)][(int)(m_observationSet.at(i).M2.val[1]*16)][(int)(m_observationSet.at(i).M2.val[2]*16)] >= 1)
			clusters[(int)(m_observationSet.at(i).M2.val[0]*16)][(int)(m_observationSet.at(i).M2.val[1]*16)][(int)(m_observationSet.at(i).M2.val[2]*16)]++;
		if (clusters[(int)(m_observationSet.at(i).Mp.val[0]*16)][(int)(m_observationSet.at(i).Mp.val[1]*16)][(int)(m_observationSet.at(i).Mp.val[2]*16)] >= 1)
			clusters[(int)(m_observationSet.at(i).Mp.val[0]*16)][(int)(m_observationSet.at(i).Mp.val[1]*16)][(int)(m_observationSet.at(i).Mp.val[2]*16)]++;
	}

	int inimg = 0;
	int inset = 0;
	for (int i = 0; i < 17; i++)
		for (int j = 0; j < 17; j++)
			for (int k = 0; k < 17; k++)
			{
				if (clusters[i][j][k] == 1)
					inimg++;
				if (clusters[i][j][k] > 1)
					inset++;
			}
	
	double ratio = (double)inset/(double)inimg;
	return ratio;
}


// generates a synthetic set of "perfect" color triples for a particular gamma curve
std::vector<ColorTriple> ObservationSetFormator::generateSyntheticSet(double gamma)
{
	srand(23);//(unsigned int)cv::getTickCount());
	std::vector<ColorTriple> synthetic;
	for (int i = 0; i < 250; i++)
	{
		ColorTriple ctn;
		Vec3d M1, M2, Mp;
		
		for (int j = 0; j < 3; j++)
		{
			M1[j] = rand() % 256;
			M2[j] = rand() % 256;
		}
		Mp = M1 + 0.5 * (M2-M1);

		ctn.M1[0] = pow(M1[0]/255., double(rand()%100)/100.);
		ctn.M1[1] = pow(M1[1]/255., double(rand()%100)/100.);
		ctn.M1[2] = pow(M1[2]/255., double(rand()%100)/100.);
		ctn.M2[0] = pow(M2[0]/255., double(rand()%100)/100.);
		ctn.M2[1] = pow(M2[1]/255., double(rand()%100)/100.);
		ctn.M2[2] = pow(M2[2]/255., double(rand()%100)/100.);
		ctn.Mp[0] = pow(Mp[0]/255., double(rand()%100)/100.);
		ctn.Mp[1] = pow(Mp[1]/255., double(rand()%100)/100.);
		ctn.Mp[2] = pow(Mp[2]/255., double(rand()%100)/100.);
		synthetic.push_back(ctn);
	}
	return synthetic;
}
