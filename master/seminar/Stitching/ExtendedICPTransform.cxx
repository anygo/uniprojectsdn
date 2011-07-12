#include "ExtendedICPTransform.h"

#include "ClosestPointFinderBruteForceCPU.h"
#include "ClosestPointFinderBruteForceGPU.h"

#include "vtkDataSet.h"
#include "vtkLandmarkTransform.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPoints.h"
#include "vtkTransform.h"
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPointData.h"

#include <complex>
#include <algorithm>
#include <QTime>
#include <QString>

vtkStandardNewMacro(ExtendedICPTransform);


extern "C"
void TransformPointsDirectlyOnGPU(double transformationMatrix[4][4], PointCoords* writeTo, float* m_Distances);

ExtendedICPTransform::ExtendedICPTransform() : vtkLinearTransform()
{
	m_Source = vtkSmartPointer<vtkPolyData>::New();
	m_Target = vtkSmartPointer<vtkPolyData>::New();
	m_Accumulate = vtkTransform::New();
	m_Accumulate->PostMultiply();
	m_SourceCoords = NULL;
	m_SourceColors = NULL;
	m_TargetCoords = NULL;
	m_TargetColors = NULL;
	m_ClosestP = NULL;
	m_Distances = NULL;
}

ExtendedICPTransform::~ExtendedICPTransform()
{
	if(m_SourceCoords) delete[] m_SourceCoords;
	if(m_SourceColors) delete[] m_SourceColors;
	if(m_TargetCoords) delete[] m_TargetCoords;
	if(m_TargetColors) delete[] m_TargetColors;
	if(m_ClosestP) delete[] m_ClosestP;
	if(m_Distances)	delete[] m_Distances;
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::Inverse()
{
	vtkSmartPointer<vtkPolyData> tmp = m_Source;
	m_Source = m_Target;
	m_Target = tmp;
	Modified();
}
vtkAbstractTransform *
ExtendedICPTransform::MakeTransform()
{
	return ExtendedICPTransform::New();
}
unsigned long int
ExtendedICPTransform::GetMTime()
{
	unsigned long result = this->vtkLinearTransform::GetMTime();
	unsigned long mtime;

	if (m_Source)
	{
		mtime = m_Source->GetMTime(); 
		if (mtime > result)
		{
			result = mtime;
		}
	}

	if (m_Target)
	{
		mtime = m_Target->GetMTime(); 
		if (mtime > result)
		{
			result = mtime;
		}
	}

	return result;
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::SetSource(vtkPolyData *source)
{
	m_Source = source;
	Modified();
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::SetTarget(vtkPolyData *target)
{
	m_Target = target;
	Modified();
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::PrintSelf(ostream& os, vtkIndent indent)
{
	os << indent << "PrintSelf() not implemented";
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::vtkPolyDataToPointCoords(vtkSmartPointer<vtkPoints> poly, PointCoords *point)
{
	for (int i = 0; i < m_NumLandmarks; ++i)
	{
		point[i].x = poly->GetPoint(static_cast<vtkIdType>(i))[0];
		point[i].y = poly->GetPoint(static_cast<vtkIdType>(i))[1];
		point[i].z = poly->GetPoint(static_cast<vtkIdType>(i))[2];
	}
}
void
ExtendedICPTransform::vtkPolyDataToPointCoordsAndColors(double percentage)
{
	int stepSource = 1;
	int stepTarget = 1;

	if (m_Source->GetNumberOfPoints() > m_NumLandmarks)
	{
		stepSource = m_Source->GetNumberOfPoints() / m_NumLandmarks;
	}
	if (m_Target->GetNumberOfPoints() > m_NumLandmarks)
	{
		stepTarget = m_Target->GetNumberOfPoints() / m_NumLandmarks;
	}

	double bounds[6];
	m_Source->GetBounds(bounds);

	// modify x, y (and z) bounds
	bounds[0] += percentage*(bounds[1] - bounds[0]);
	bounds[1] -= percentage*(bounds[1] - bounds[0]);
	bounds[2] += percentage*(bounds[3] - bounds[2]);
	bounds[3] -= percentage*(bounds[3] - bounds[2]);
	bounds[4] += percentage*(bounds[5] - bounds[4]);
	bounds[5] -= percentage*(bounds[5] - bounds[4]);

	vtkDataArray* scalarsPtr;
	PointCoords* curCoords;
	PointColors* curColors;

	scalarsPtr = m_Source->GetPointData()->GetScalars();
	for (int i = 0, j = 0; i < m_NumLandmarks; ++i, j = (j + stepSource) % m_Source->GetNumberOfPoints())
	{
		curCoords = &m_SourceCoords[i];
		double* curPoint = m_Source->GetPoint(static_cast<vtkIdType>(j));
		curCoords->x = curPoint[0];
		curCoords->y = curPoint[1];
		curCoords->z = curPoint[2];

		if (!(  curCoords->x >= bounds[0] && curCoords->x <= bounds[1] &&
				curCoords->y >= bounds[2] && curCoords->y <= bounds[3] &&
				curCoords->z >= bounds[4] && curCoords->z <= bounds[5]
			))
		{
			--i;
			continue;
		}

		// conversion from RGB to rgb (r = R/(R+G+B), ...)

		double* tuple = scalarsPtr->GetTuple(static_cast<vtkIdType>(j));
		float r_g_b = tuple[0] + tuple[1] + tuple[2];

		float factor = m_NormalizeRGBToDistanceValuesFactor / std::max(r_g_b, 1.f);

		curColors = &m_SourceColors[i];

		curColors->r = tuple[0] * factor;
		curColors->g = tuple[1] * factor;
		curColors->b = tuple[2] * factor;
	}

	scalarsPtr = m_Target->GetPointData()->GetScalars();
	for (int i = 0, j = 0; i < m_NumLandmarks; ++i, j += stepTarget)
	{
		curCoords = &m_TargetCoords[i];
		double* curPoint = m_Target->GetPoint(static_cast<vtkIdType>(j));
		curCoords->x = curPoint[0];
		curCoords->y = curPoint[1];
		curCoords->z = curPoint[2];

		// conversion from RGB to rgb (r = R/(R+G+B), ...)
		double* tuple = scalarsPtr->GetTuple(static_cast<vtkIdType>(j));
		float r_g_b = tuple[0] + tuple[1] + tuple[2];

		float factor = m_NormalizeRGBToDistanceValuesFactor / std::max(r_g_b, 1.f);

		curColors = &m_TargetColors[i];

		curColors->r = tuple[0] * factor;
		curColors->g = tuple[1] * factor;
		curColors->b = tuple[2] * factor;
	}
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::InternalUpdate() 
{
	// configure ClosestPointFinder
	m_ClosestPointFinder->SetTarget(m_TargetCoords, m_TargetColors, m_SourceCoords, m_SourceColors);

	m_Accumulate->Identity();

	// apply previous transform
	if (m_ApplyPreviousTransform)
	{
		TransformPointsDirectlyOnGPU(m_PreviousTransformationMatrix->Element, m_SourceCoords, NULL);
		m_Accumulate->Concatenate(m_PreviousTransformationMatrix);
	}

	vtkSmartPointer<vtkMatrix4x4> mat = vtkSmartPointer<vtkMatrix4x4>::New();

	unsigned short* indices;
	float totaldist;
	m_NumIter = 0;

//#define RUNTIME_EVALUATION
#ifdef RUNTIME_EVALUATION
	const int RUNTIME_ITER = 100;
	double RUNTIMES_ELAPSED[3] = {0,0,0};
	QTime T_RUNTIME;
#endif

	while (true)
	{
		// Set locators source points and perfom nearest neighbor search
#ifdef RUNTIME_EVALUATION
		T_RUNTIME.start();
		for (int runtimeIteration = 0; runtimeIteration < RUNTIME_ITER; ++runtimeIteration)
		{
#endif
			indices = m_ClosestPointFinder->FindClosestPoints(m_SourceCoords, m_SourceColors);

#ifdef RUNTIME_EVALUATION
		}
		RUNTIMES_ELAPSED[0] += T_RUNTIME.elapsed();
#endif

		for(int i = 0; i < m_NumLandmarks; ++i)
		{
			int index = indices[i];
			m_ClosestP[i].x = m_TargetCoords[index].x;
			m_ClosestP[i].y = m_TargetCoords[index].y;
			m_ClosestP[i].z = m_TargetCoords[index].z;
		}

#ifdef RUNTIME_EVALUATION
		T_RUNTIME.start();
		for (int runtimeIteration = 0; runtimeIteration < RUNTIME_ITER; ++runtimeIteration)
		{
#endif
			mat->DeepCopy(EstimateTransformationMatrix(m_SourceCoords, m_ClosestP));

#ifdef RUNTIME_EVALUATION
		}
		RUNTIMES_ELAPSED[1] += T_RUNTIME.elapsed();
#endif		

		// concatenate transformation matrices
		m_Accumulate->Concatenate(mat);

		++m_NumIter;
		if (m_NumIter >= m_MaxIter) 
			break;

		// move mesh and compute mean distance to previous iteration
		totaldist = 0.f;

		// transform on gpu
#ifdef RUNTIME_EVALUATION
		vtkMatrix4x4* matInv = vtkMatrix4x4::New();
		vtkMatrix4x4::Invert(mat, matInv);
		T_RUNTIME.start();		
		for (int runtimeIteration = 0; runtimeIteration < RUNTIME_ITER / 2; ++runtimeIteration)
		{
#endif
			TransformPointsDirectlyOnGPU(mat->Element, m_SourceCoords, m_Distances);
			
#ifdef RUNTIME_EVALUATION
			TransformPointsDirectlyOnGPU(matInv->Element, m_SourceCoords, m_Distances);	
		}
		TransformPointsDirectlyOnGPU(mat->Element, m_SourceCoords, m_Distances);
		RUNTIMES_ELAPSED[2] += T_RUNTIME.elapsed();
#endif

		/*for(int i = 0; i < m_NumLandmarks; i++)
		{
			totaldist += m_Distances[i];
		}	
		
		m_MeanDist = totaldist / (float)m_NumLandmarks;

		if (m_MeanDist <= m_MaxMeanDist)
			break;*/
	} 

	// now recover accumulated result
	this->Matrix->DeepCopy(m_Accumulate->GetMatrix());

#ifdef RUNTIME_EVALUATION
	std::cout << "Runtime Evaluation:" << std::endl;
	double RUNTIME_OVERALL = RUNTIMES_ELAPSED[0] + RUNTIMES_ELAPSED[1] + RUNTIMES_ELAPSED[2];
	std::cout << "step0: " << (double)RUNTIMES_ELAPSED[0] / (double)(RUNTIME_ITER*m_NumIter) << " | " << (RUNTIMES_ELAPSED[0]*100) / RUNTIME_OVERALL << " %" << std::endl;
	std::cout << "step1: " << (double)RUNTIMES_ELAPSED[1] / (double)(RUNTIME_ITER*m_NumIter) << " | " << (RUNTIMES_ELAPSED[1]*100) / RUNTIME_OVERALL << " %" << std::endl;
	std::cout << "step2: " << (double)RUNTIMES_ELAPSED[2] / (double)(RUNTIME_ITER*m_NumIter) << " | " << (RUNTIMES_ELAPSED[2]*100) / RUNTIME_OVERALL << " %" << std::endl;
	std::cout << std::endl;

	std::cout << EVAL_FILENAME << std::endl;
	std::ofstream file(EVAL_FILENAME, ios::app);

	file << m_NumLandmarks << " " << (RUNTIMES_ELAPSED[0]*100) / RUNTIME_OVERALL << " " << (RUNTIMES_ELAPSED[1]*100) / RUNTIME_OVERALL << " " << (RUNTIMES_ELAPSED[2]*100) / RUNTIME_OVERALL << " " <<
		(double)RUNTIMES_ELAPSED[0] / (double)(RUNTIME_ITER*m_NumIter) << " " << (double)RUNTIMES_ELAPSED[1] / (double)(RUNTIME_ITER*m_NumIter) << " " << (double)RUNTIMES_ELAPSED[2] / (double)(RUNTIME_ITER*m_NumIter) << std::endl;
#endif
}

vtkMatrix4x4*
ExtendedICPTransform::EstimateTransformationMatrix(PointCoords* source, PointCoords* target)
{
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