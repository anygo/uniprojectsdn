#include "ExtendedICPTransform.h"

#include "ClosestPointFinderBruteForceCPU.h"
//#include "CudaTest.cu"

#include "vtkDataSet.h"
#include "vtkLandmarkTransform.h"
#include "vtkMath.h"
#include "vtkObjectFactory.h"
#include "vtkPoints.h"
#include "vtkTransform.h"
#include "vtkSmartPointer.h"
#include "vtkPolyData.h"

#include <complex>

vtkStandardNewMacro(ExtendedICPTransform);


//----------------------------------------------------------------------------
ExtendedICPTransform::ExtendedICPTransform() : vtkLinearTransform()
{
	m_Source = vtkSmartPointer<vtkPolyData>::New();
	m_Target = vtkSmartPointer<vtkPolyData>::New();
	m_LandmarkTransform = vtkSmartPointer<vtkLandmarkTransform>::New();
	m_MaxIter = 500;
	m_MaxMeanDist = 0.0001;
	m_MaxLandmarks = 1000;

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
ExtendedICPTransform::vtkPolyDataToPoint6DArray(vtkSmartPointer<vtkPoints> poly, Point6D *point)
{
	for (int i = 0; i < m_MaxLandmarks; ++i)
	{
		point[i].x = poly->GetPoint(static_cast<vtkIdType>(i))[0];
		point[i].y = poly->GetPoint(static_cast<vtkIdType>(i))[1];
		point[i].z = poly->GetPoint(static_cast<vtkIdType>(i))[2];
	}
}
void
ExtendedICPTransform::vtkPolyDataToPoint6DArray()
{

	int stepSource = 1;
	int stepTarget = 1;

	if (m_Source->GetNumberOfPoints() > m_MaxLandmarks)
	{
		stepSource = m_Source->GetNumberOfPoints() / m_MaxLandmarks;
	}
	if (m_Target->GetNumberOfPoints() > m_MaxLandmarks)
	{
		stepTarget = m_Target->GetNumberOfPoints() / m_MaxLandmarks;
	}

	m_SourcePoints = new Point6D[m_MaxLandmarks];
	m_TargetPoints = new Point6D[m_MaxLandmarks];

	for (int i = 0, j = 0; i < m_MaxLandmarks; ++i, j += stepSource)
	{
		m_SourcePoints[i].x = m_Source->GetPoint(static_cast<vtkIdType>(j))[0];
		m_SourcePoints[i].y = m_Source->GetPoint(static_cast<vtkIdType>(j))[1];
		m_SourcePoints[i].z = m_Source->GetPoint(static_cast<vtkIdType>(j))[2];
	}

	for (int i = 0, j = 0; i < m_MaxLandmarks; ++i, j += stepTarget)
	{
		m_TargetPoints[i].x = m_Target->GetPoint(static_cast<vtkIdType>(j))[0];
		m_TargetPoints[i].y = m_Target->GetPoint(static_cast<vtkIdType>(j))[1];
		m_TargetPoints[i].z = m_Target->GetPoint(static_cast<vtkIdType>(j))[2];
	}
}
//----------------------------------------------------------------------------
void
ExtendedICPTransform::InternalUpdate() 
{
	// Test Cuda
	//cudaTest();

	// transform vtkPolyData in our own structures
	vtkPolyDataToPoint6DArray();

	// create own locator
	ClosestPointFinderBruteForceCPU cp_locator(m_MaxLandmarks);
	// set target points once
	cp_locator.SetTarget(m_TargetPoints);
	// set used distance metric
	cp_locator.SetMetric(m_Metric);

	// Allocate some points.
	vtkSmartPointer<vtkPoints> points1 =
		vtkSmartPointer<vtkPoints>::New();
	points1->SetNumberOfPoints(m_MaxLandmarks);

	vtkSmartPointer<vtkPoints> closestp =
		vtkSmartPointer<vtkPoints>::New();
	closestp->SetNumberOfPoints(m_MaxLandmarks);

	vtkSmartPointer<vtkPoints> points2 =
		vtkSmartPointer<vtkPoints>::New();
	points2->SetNumberOfPoints(m_MaxLandmarks);

	vtkSmartPointer<vtkTransform> accumulate =
		vtkTransform::New();
	accumulate->PostMultiply();

	double p1[3], p2[3];

	for (int i = 0; i < m_MaxLandmarks; i++)
	{
		points1->SetPoint(static_cast<vtkIdType>(i), m_SourcePoints[i].x, m_SourcePoints[i].y, m_SourcePoints[i].z);
	}

	// go
	vtkSmartPointer<vtkPoints> temp;
	vtkSmartPointer<vtkPoints> a = points1;
	vtkSmartPointer<vtkPoints> b = points2;

	double totaldist;

	m_NumIter = 0;

	while (true)
	{
		// Set locators source points and perfom nearest neighbor search
		int* indices = cp_locator.FindClosestPoints( m_SourcePoints ); // FindClosestPoints(m_SourcePoints, m_TargetPoints);
		for(int i = 0; i < m_MaxLandmarks; ++i)
		{
			int index = indices[i];
			closestp->SetPoint(i, m_TargetPoints[index].x, m_TargetPoints[index].y, m_TargetPoints[index].z );
		}

		// build the landmark transform
		m_LandmarkTransform->SetSourceLandmarks(a);
		m_LandmarkTransform->SetTargetLandmarks(closestp);
		m_LandmarkTransform->Update();

		// concatenate
		accumulate->Concatenate(m_LandmarkTransform->GetMatrix());

		m_NumIter++;
		if (m_NumIter >= m_MaxIter) 
		{
			break;
		}

		// move mesh and compute mean distance
		totaldist = 0.0;

		for(int i = 0; i < m_MaxLandmarks; i++)
		{
			a->GetPoint(i, p1);
			m_LandmarkTransform->InternalTransformPoint(p1, p2);
			b->SetPoint(i, p2);

			switch (m_Metric)
			{
			case ABSOLUTE_DISTANCE:
				totaldist += std::abs(p1[0] - p2[0]) + std::abs(p1[1] - p2[1]) + std::abs(p1[2] - p2[2]); break;
			case LOG_ABSOLUTE_DISTANCE:
				totaldist += std::log(std::abs(p1[0] - p2[0]) + std::abs(p1[1] - p2[1]) + std::abs(p1[2] - p2[2]) + 1.0); break;
			case SQUARED_DISTANCE:
				totaldist += vtkMath::Distance2BetweenPoints(p1, p2);
			}

		}

		m_MeanDist = totaldist / (double)m_MaxLandmarks;
		std::cout << "\rIteration " << m_NumIter << ":\t mean distance = " << m_MeanDist << "           ";
			
		if (m_MeanDist <= m_MaxMeanDist)
		{
			break;
		}

		// swapping
		temp = a;
		a = b;
		b = temp;

		vtkPolyDataToPoint6DArray( a, m_SourcePoints );
	} 

	std::cout << std::endl;

	// now recover accumulated result
	this->Matrix->DeepCopy(accumulate->GetMatrix());
}
//----------------------------------------------------------------------------
