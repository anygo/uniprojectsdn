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
	m_LandmarkTransform = vtkSmartPointer<vtkLandmarkTransform>::New();
	m_SourceCoords = NULL;
	m_SourceColors = NULL;
	m_TargetCoords = NULL;
	m_TargetColors = NULL;
	m_Distances = NULL;
}

ExtendedICPTransform::~ExtendedICPTransform()
{
	if(m_SourceCoords) delete[] m_SourceCoords;
	if(m_SourceColors) delete[] m_SourceColors;
	if(m_TargetCoords) delete[] m_TargetCoords;
	if(m_TargetColors) delete[] m_TargetColors;
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

		float factor = m_NormalizeRGBToDistanceValuesFactor / std::max(r_g_b, FLT_EPSILON);

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

		float factor = m_NormalizeRGBToDistanceValuesFactor / std::max(r_g_b, FLT_EPSILON);

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
	// for some reason, we need this
	m_Points1->Modified();
	m_Points2->Modified();
	m_Closestp->Modified();

	// configure ClosestPointFinder
	QTime ts;
	ts.start();
	m_ClosestPointFinder->SetTarget(m_TargetCoords, m_TargetColors, m_SourceCoords, m_SourceColors);
	//std::cout << "SetTarget() " << ts.elapsed() << " ms" << std::endl;


	vtkSmartPointer<vtkTransform> accumulate =
		vtkTransform::New();
	accumulate->PostMultiply();

	// apply previous transform
	if (m_ApplyPreviousTransform)
	{
		TransformPointsDirectlyOnGPU(m_PreviousTransformationMatrix->Element, m_SourceCoords, NULL);
		accumulate->Concatenate(m_PreviousTransformationMatrix);
	}

	double p1[3], p2[3];
	unsigned short* indices;

	for (int i = 0; i < m_NumLandmarks; i++)
	{
		m_Points1->SetPoint(static_cast<vtkIdType>(i), m_SourceCoords[i].x, m_SourceCoords[i].y, m_SourceCoords[i].z);
	}

	// go
	vtkSmartPointer<vtkPoints> a2;
	if (m_RemoveOutliers && m_OutlierRate > 0.0)
	{
		a2 = vtkSmartPointer<vtkPoints>::New();
	}

	vtkSmartPointer<vtkPoints> temp;
	vtkSmartPointer<vtkPoints> a = m_Points1;
	vtkSmartPointer<vtkPoints> b = m_Points2;

	float totaldist;
	m_NumIter = 0;

	QTime findTime;
	int findTimeElapsed = 0;
	while (true)
	{
		// Set locators source points and perfom nearest neighbor search
		findTime.start();
		indices = m_ClosestPointFinder->FindClosestPoints(m_SourceCoords, m_SourceColors);
		findTimeElapsed += findTime.elapsed();

		if (m_RemoveOutliers && m_OutlierRate > 0.0)
		{
			std::vector<float> sortedDistances(m_NumLandmarks);
			float* dists = m_ClosestPointFinder->GetDistances();
			m_MeanTargetDistance = 0;
			for(int i = 0; i < m_NumLandmarks; ++i)
			{
				sortedDistances[i] = dists[i];
				m_MeanTargetDistance += (dists[i]);
			}
			m_MeanTargetDistance /= static_cast<float>(m_NumLandmarks);

			std::sort(sortedDistances.begin(), sortedDistances.end());

			int thresholdIdx = floor((1.0 - m_OutlierRate) * static_cast<float>(m_NumLandmarks - 1));
			float threshold = sortedDistances[thresholdIdx];

			int number = thresholdIdx + 1;

			// perfect match?
			if (threshold < FLT_EPSILON)
			{
				threshold = FLT_MAX;
				number = m_NumLandmarks;
			}

			// calling Modified() is necessary otherwise object properties won't change
			m_Closestp->SetNumberOfPoints(number);
			a2->SetNumberOfPoints(number);
			m_Closestp->Modified();
			a2->Modified();

			int count = 0;
			for(int i = 0; i < m_NumLandmarks; ++i)
			{
				if(dists[i] <= threshold) 
				{
					int index = indices[i];
					m_Closestp->SetPoint(count, m_TargetCoords[index].x, m_TargetCoords[index].y, m_TargetCoords[index].z);
					a2->SetPoint(count, m_SourceCoords[i].x, m_SourceCoords[i].y, m_SourceCoords[i].z);
					++count;
				}
			}
			m_LandmarkTransform->SetSourceLandmarks(a2);

		} else
		{
			for(int i = 0; i < m_NumLandmarks; ++i)
			{
				int index = indices[i];
				m_Closestp->SetPoint(i, m_TargetCoords[index].x, m_TargetCoords[index].y, m_TargetCoords[index].z);
			}
			m_LandmarkTransform->SetSourceLandmarks(a);
		}

		// build the landmark transform
		m_LandmarkTransform->SetTargetLandmarks(m_Closestp);
		m_LandmarkTransform->Update();

		// concatenate transformation matrices
		accumulate->Concatenate(m_LandmarkTransform->GetMatrix());

		++m_NumIter;
		if (m_NumIter >= m_MaxIter) 
		{
			break;
		}

		// move mesh and compute mean distance to previous iteration
		totaldist = 0.f;

		// transform on gpu
		if (m_ClosestPointFinder->usesGPU())
		{	
			TransformPointsDirectlyOnGPU(m_LandmarkTransform->GetMatrix()->Element, m_SourceCoords, m_Distances);
			for(int i = 0; i < m_NumLandmarks; i++)
			{
				totaldist += m_Distances[i];
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
		DBG << "\rICP Iteration " << m_NumIter << ":\t mean distance = " << m_MeanDist << "\t\t";

		if (m_MeanDist <= m_MaxMeanDist)
		{
			break;
		}

		// swap
		temp = a;
		a = b;
		b = temp;

		if (!m_ClosestPointFinder->usesGPU())
		{
			vtkPolyDataToPointCoords(a, m_SourceCoords);
		}
	} 
	DBG << std::endl;

	//std::cout << "avg findTimeElapsed: " << static_cast<double>(findTimeElapsed) / static_cast<double>(m_NumIter) << std::endl;

	// now recover accumulated result
	this->Matrix->DeepCopy(accumulate->GetMatrix());

}