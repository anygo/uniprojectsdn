#include "ExtendedICPTransform.h"

#include "vtkDataSet.h"
#include "vtkLandmarkTransform.h"
#include "vtkMath.h"
#include "vtkPoints.h"
#include "vtkTransform.h"
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"

#include <complex>
#include <algorithm>
#include <QTime>
#include <QString>


extern "C"
void CUDATransformPoints(double transformationMatrix[4][4], float4* toBeTransformed, int numPoints, float* distances);

ExtendedICPTransform::ExtendedICPTransform()
{
	//std::cout << "ExtendedICPTransform" << std::endl;

	m_Accumulate = vtkTransform::New();
	m_Accumulate->PostMultiply();
	m_Source = NULL;
	m_Target = NULL;
	m_ClosestP = NULL;
	m_Distances = NULL;
	m_Indices = NULL;
	m_devDistances = NULL;
}

ExtendedICPTransform::~ExtendedICPTransform()
{
	//std::cout << "~ExtendedICPTransform" << std::endl;
	if(m_Source) delete[] m_Source;
	if(m_Target) delete[] m_Target;
	if(m_ClosestP) delete[] m_ClosestP;
	if(m_Distances)	delete[] m_Distances;
	if(m_Indices) delete[] m_Indices;
	if(m_devDistances) cudaFree(m_devDistances);
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::SetSource(float4 *source)
{
	//std::cout << "SetSource" << std::endl;
	m_devSource = source;
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::SetTarget(float4 *target)
{
	//std::cout << "SetTarget" << std::endl;
	m_devTarget = target;
}
//----------------------------------------------------------------------------
vtkMatrix4x4*
ExtendedICPTransform::StartICP() 
{
	//std::cout << "StartICP" << std::endl;

	// configure ClosestPointFinder
	m_ClosestPointFinder->Initialize( m_devTarget, NULL, m_devSource, NULL);

	m_Accumulate->Identity();

	float totaldist;
	m_NumIter = 0;

	int findTimeElapsed = 0;

	// copy target landmarks only once
	cutilSafeCall(cudaMemcpy(m_Target, m_devTarget, m_NumLandmarks*sizeof(float4), cudaMemcpyDeviceToHost));

	while (true)
	{
		cutilSafeCall(cudaMemcpy(m_Source, m_devSource, m_NumLandmarks*sizeof(float4), cudaMemcpyDeviceToHost));
		
		// Set locators source points and perfom nearest neighbor search
		
		m_ClosestPointFinder->FindClosestPoints(m_Indices, m_Distances);

		for(int i = 0; i < m_NumLandmarks; ++i)
		{
			int index = m_Indices[i];
			m_ClosestP[i] = m_Target[index];
		}

		vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();
		mat->DeepCopy(EstimateTransformationMatrix(m_Source, m_ClosestP));

		// concatenate transformation matrices
		m_Accumulate->Concatenate(mat);

		++m_NumIter;
		if (m_NumIter >= m_MaxIter) 
			break;

		// move mesh and compute mean distance to previous iteration
		totaldist = 0.f;

		// transform on gpu
		CUDATransformPoints(mat->Element, m_devSource, m_NumLandmarks, m_devDistances);

		cutilSafeCall(cudaMemcpy(m_Distances, m_devDistances, m_NumLandmarks*sizeof(float), cudaMemcpyDeviceToHost));

		for(int i = 0; i < m_NumLandmarks; i++)
		{
			totaldist += m_Distances[i];
		}	

		m_MeanDist = totaldist / (float)m_NumLandmarks;

		if (m_MeanDist <= m_MaxMeanDist)
			break;
	} 

	// now recover accumulated result
	return m_Accumulate->GetMatrix();
}

vtkMatrix4x4*
ExtendedICPTransform::EstimateTransformationMatrix(float4* source, float4* target)
{
	//std::cout << "EstimateTransformationMatrix" << std::endl;
	int i;

	vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();

	// -- find the centroid of each set --
	double source_centroid[3] = {0,0,0};
	double target_centroid[3] = {0,0,0};
	for(i = 0; i < m_NumLandmarks; ++i)
	{
		source_centroid[0] += source[i].x;
		source_centroid[1] += source[i].y;
		source_centroid[2] += source[i].z;
		target_centroid[0] += target[i].x;
		target_centroid[1] += target[i].y;
		target_centroid[2] += target[i].z;
	}
	source_centroid[0] /= m_NumLandmarks;
	source_centroid[1] /= m_NumLandmarks;
	source_centroid[2] /= m_NumLandmarks;
	target_centroid[0] /= m_NumLandmarks;
	target_centroid[1] /= m_NumLandmarks;
	target_centroid[2] /= m_NumLandmarks;

	// -- build the 3x3 matrix M --
	double M[3][3];
	double AAT[3][3];
	for(i = 0; i < 3; ++i) 
	{
		AAT[i][0] = M[i][0]=0.0F; // fill M with zeros
		AAT[i][1] = M[i][1]=0.0F; 
		AAT[i][2] = M[i][2]=0.0F; 
	}

	int pt;
	double a[3], b[3];
	double sa = 0.0F, sb = 0.0F;
	for(pt = 0; pt < m_NumLandmarks; ++pt)
	{
		// get the origin-centred point (a) in the source set
		a[0] = source[pt].x;
		a[1] = source[pt].y;
		a[2] = source[pt].z;
		a[0] -= source_centroid[0];
		a[1] -= source_centroid[1];
		a[2] -= source_centroid[2];
		// get the origin-centred point (b) in the target set
		b[0] = target[pt].x;
		b[1] = target[pt].y;
		b[2] = target[pt].z;
		b[0] -= target_centroid[0];
		b[1] -= target_centroid[1];;
		b[2] -= target_centroid[2];;
		// accumulate the products a*T(b) into the matrix M
		for(i = 0; i < 3; ++i) 
		{
			M[i][0] += a[i]*b[0];
			M[i][1] += a[i]*b[1];
			M[i][2] += a[i]*b[2];
		}
		// accumulate scale factors (if desired)
		sa += a[0]*a[0]+a[1]*a[1]+a[2]*a[2];
		sb += b[0]*b[0]+b[1]*b[1]+b[2]*b[2];
	}

	// compute required scaling factor (if desired)
	double scale = (double)sqrt(sb/sa);

	// -- build the 4x4 matrix N --
	double Ndata[4][4];
	double *N[4];
	for(i=0;i<4;i++)
	{
		N[i] = Ndata[i];
		N[i][0]=0.0F; // fill N with zeros
		N[i][1]=0.0F;
		N[i][2]=0.0F;
		N[i][3]=0.0F;
	}
	// on-diagonal elements
	N[0][0] = M[0][0]+M[1][1]+M[2][2];
	N[1][1] = M[0][0]-M[1][1]-M[2][2];
	N[2][2] = -M[0][0]+M[1][1]-M[2][2];
	N[3][3] = -M[0][0]-M[1][1]+M[2][2];
	// off-diagonal elements
	N[0][1] = N[1][0] = M[1][2]-M[2][1];
	N[0][2] = N[2][0] = M[2][0]-M[0][2];
	N[0][3] = N[3][0] = M[0][1]-M[1][0];

	N[1][2] = N[2][1] = M[0][1]+M[1][0];
	N[1][3] = N[3][1] = M[2][0]+M[0][2];
	N[2][3] = N[3][2] = M[1][2]+M[2][1];

	// -- eigen-decompose N (is symmetric) --
	double eigenvectorData[4][4];
	double *eigenvectors[4],eigenvalues[4];

	eigenvectors[0] = eigenvectorData[0];
	eigenvectors[1] = eigenvectorData[1];
	eigenvectors[2] = eigenvectorData[2];
	eigenvectors[3] = eigenvectorData[3];

	vtkMath::JacobiN(N,4,eigenvalues,eigenvectors);

	// the eigenvector with the largest eigenvalue is the quaternion we want
	// (they are sorted in decreasing order for us by JacobiN)
	double w,x,y,z;

	// points are not collinear
	w = eigenvectors[0][0];
	x = eigenvectors[1][0];
	y = eigenvectors[2][0];
	z = eigenvectors[3][0];

	// convert quaternion to a rotation matrix
	double ww = w*w;
	double wx = w*x;
	double wy = w*y;
	double wz = w*z;

	double xx = x*x;
	double yy = y*y;
	double zz = z*z;

	double xy = x*y;
	double xz = x*z;
	double yz = y*z;

	mat->Element[0][0] = ww + xx - yy - zz; 
	mat->Element[1][0] = 2.0*(wz + xy);
	mat->Element[2][0] = 2.0*(-wy + xz);

	mat->Element[0][1] = 2.0*(-wz + xy);  
	mat->Element[1][1] = ww - xx + yy - zz;
	mat->Element[2][1] = 2.0*(wx + yz);

	mat->Element[0][2] = 2.0*(wy + xz);
	mat->Element[1][2] = 2.0*(-wx + yz);
	mat->Element[2][2] = ww - xx - yy + zz;

	// the translation is given by the difference in the transformed source
	// centroid and the target centroid
	double sx, sy, sz;

	sx = mat->Element[0][0] * source_centroid[0] +
		mat->Element[0][1] * source_centroid[1] +
		mat->Element[0][2] * source_centroid[2];
	sy = mat->Element[1][0] * source_centroid[0] +
		mat->Element[1][1] * source_centroid[1] +
		mat->Element[1][2] * source_centroid[2];
	sz = mat->Element[2][0] * source_centroid[0] +
		mat->Element[2][1] * source_centroid[1] +
		mat->Element[2][2] * source_centroid[2];

	mat->Element[0][3] = target_centroid[0] - sx;
	mat->Element[1][3] = target_centroid[1] - sy;
	mat->Element[2][3] = target_centroid[2] - sz;

	// fill the bottom row of the 4x4 matrix
	mat->Element[3][0] = 0.0;
	mat->Element[3][1] = 0.0;
	mat->Element[3][2] = 0.0;
	mat->Element[3][3] = 1.0;

	mat->Modified();

	return mat;
}