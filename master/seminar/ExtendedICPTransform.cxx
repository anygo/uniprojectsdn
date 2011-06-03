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

vtkStandardNewMacro(ExtendedICPTransform);


extern "C"
void TransformPointsDirectlyOnGPU(int nrOfPoints, double transformationMatrix[4][4], PointCoords* writeTo, float* distances);

//----------------------------------------------------------------------------
ExtendedICPTransform::ExtendedICPTransform() : vtkLinearTransform()
{
	m_Source = vtkSmartPointer<vtkPolyData>::New();
	m_Target = vtkSmartPointer<vtkPolyData>::New();
	m_LandmarkTransform = vtkSmartPointer<vtkLandmarkTransform>::New();
	m_MaxIter = 500;
	m_MaxMeanDist = 0.0001;
	m_NumLandmarks = 1000;

	m_NumIter = 0;
	m_MeanDist = 0.0;
}

ExtendedICPTransform::~ExtendedICPTransform() {}
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

	if (m_LandmarkTransform)
	{
		mtime = m_LandmarkTransform->GetMTime();
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
ExtendedICPTransform::vtkPolyDataToPointCoordsAndColors()
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

	m_SourceCoords = new PointCoords[m_NumLandmarks];
	m_SourceColors = new PointColors[m_NumLandmarks];
	m_TargetCoords = new PointCoords[m_NumLandmarks];
	m_TargetColors = new PointColors[m_NumLandmarks];

	for (int i = 0, j = 0; i < m_NumLandmarks; ++i, j += stepSource)
	{
		m_SourceCoords[i].x = m_Source->GetPoint(static_cast<vtkIdType>(j))[0];
		m_SourceCoords[i].y = m_Source->GetPoint(static_cast<vtkIdType>(j))[1];
		m_SourceCoords[i].z = m_Source->GetPoint(static_cast<vtkIdType>(j))[2];
		m_SourceColors[i].r = m_Source->GetPointData()->GetScalars()->GetTuple(static_cast<vtkIdType>(j))[0];
		m_SourceColors[i].g = m_Source->GetPointData()->GetScalars()->GetTuple(static_cast<vtkIdType>(j))[1];
		m_SourceColors[i].b = m_Source->GetPointData()->GetScalars()->GetTuple(static_cast<vtkIdType>(j))[2];
	}

	for (int i = 0, j = 0; i < m_NumLandmarks; ++i, j += stepTarget)
	{
		m_TargetCoords[i].x = m_Target->GetPoint(static_cast<vtkIdType>(j))[0];
		m_TargetCoords[i].y = m_Target->GetPoint(static_cast<vtkIdType>(j))[1];
		m_TargetCoords[i].z = m_Target->GetPoint(static_cast<vtkIdType>(j))[2];
		m_TargetColors[i].r = m_Target->GetPointData()->GetScalars()->GetTuple(static_cast<vtkIdType>(j))[0];
		m_TargetColors[i].g = m_Target->GetPointData()->GetScalars()->GetTuple(static_cast<vtkIdType>(j))[1];
		m_TargetColors[i].b = m_Target->GetPointData()->GetScalars()->GetTuple(static_cast<vtkIdType>(j))[2];
	}
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::InternalUpdate() 
{

	// transform vtkPolyData in our own structures
	vtkPolyDataToPointCoordsAndColors();

	// configure ClosestPointFinder
	m_ClosestPointFinder->SetTarget(m_TargetCoords, m_TargetColors);
	
	// allocate some points used for icp
	vtkSmartPointer<vtkPoints> points1 =
		vtkSmartPointer<vtkPoints>::New();
	points1->SetNumberOfPoints(m_NumLandmarks);

	vtkSmartPointer<vtkPoints> closestp =
		vtkSmartPointer<vtkPoints>::New();
	closestp->SetNumberOfPoints(m_NumLandmarks);

	vtkSmartPointer<vtkPoints> points2 =
		vtkSmartPointer<vtkPoints>::New();
	points2->SetNumberOfPoints(m_NumLandmarks);

	vtkSmartPointer<vtkTransform> accumulate =
		vtkTransform::New();
	accumulate->PostMultiply();

	double p1[3], p2[3];
	// for gpu based distance computation
	float* distances = new float[m_NumLandmarks];

	for (int i = 0; i < m_NumLandmarks; i++)
	{
		points1->SetPoint(static_cast<vtkIdType>(i), m_SourceCoords[i].x, m_SourceCoords[i].y, m_SourceCoords[i].z);
	}

	// go
	vtkSmartPointer<vtkPoints> temp;
	vtkSmartPointer<vtkPoints> a = points1;
	vtkSmartPointer<vtkPoints> b = points2;

	float totaldist;
	m_NumIter = 0;

	while (true)
	{
		// Set locators source points and perfom nearest neighbor search
		unsigned short* indices = m_ClosestPointFinder->FindClosestPoints(m_SourceCoords, m_SourceColors);
		for(int i = 0; i < m_NumLandmarks; ++i)
		{
			int index = indices[i];
			closestp->SetPoint(i, m_TargetCoords[index].x, m_TargetCoords[index].y, m_TargetCoords[index].z );
		}

		// build the landmark transform
		m_LandmarkTransform->SetSourceLandmarks(a);
		m_LandmarkTransform->SetTargetLandmarks(closestp);
		m_LandmarkTransform->Update();

		// concatenate transformation matrices
		accumulate->Concatenate(m_LandmarkTransform->GetMatrix());

		m_NumIter++;
		if (m_NumIter >= m_MaxIter) 
		{
			break;
		}

		// move mesh and compute mean distance to previous iteration
		totaldist = 0.f;

		// transform on gpu
		if (m_ClosestPointFinder->usesGPU())
		{	
			TransformPointsDirectlyOnGPU(m_NumLandmarks, m_LandmarkTransform->GetMatrix()->Element, m_SourceCoords, distances);
			for(int i = 0; i < m_NumLandmarks; i++)
			{
				totaldist += distances[i];
				b->SetPoint(i, m_SourceCoords[i].x, m_SourceCoords[i].y, m_SourceCoords[i].z);
			}	

		} else
		{
			for(int i = 0; i < m_NumLandmarks; i++)
			{
				a->GetPoint(i, p1);
				m_LandmarkTransform->InternalTransformPoint(p1, p2);
				b->SetPoint(i, p2);

				totaldist += vtkMath::Distance2BetweenPoints(p1, p2);
			}
		}

		m_MeanDist = totaldist / (float)m_NumLandmarks;
		std::cout << "\rICP Iteration " << m_NumIter << ":\t mean distance = " << m_MeanDist << "\t\t";
			
		if (m_MeanDist <= m_MaxMeanDist)
		{
			break;
		}

		// swapping
		temp = a;
		a = b;
		b = temp;


		if (!m_ClosestPointFinder->usesGPU()) {
			vtkPolyDataToPointCoords(a, m_SourceCoords);
		}
	} 

	std::cout << std::endl;

	// now recover accumulated result
	this->Matrix->DeepCopy(accumulate->GetMatrix());

	// cleanup data structure for gpu version
	delete [] distances;
}