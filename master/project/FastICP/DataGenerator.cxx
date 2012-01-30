#include "DataGenerator.h"
#include "ritkLUTData.h"
#include "defs.h"

#include <algorithm>
#include <iterator>

#include "vtkSmartPointer.h"
#include "vtkMath.h"
#include "vtkTransform.h"


//----------------------------------------------------------------------------
extern "C"
void CUDATransformPoints3D(float* points, float* m, unsigned int numPts, unsigned int dim);
//----------------------------------------------------------------------------


//----------------------------------------------------------------------------
DataGenerator::DataGenerator()
{
	// Init number of points
	m_NumPts = 2048;

	// Init container for set of fixed points
	m_FixedPts = DatasetContainer::New();
	DatasetContainer::SizeType DataSize;
	DataSize.SetElement(0, ICP_DATA_DIM*m_NumPts);
	m_FixedPts->SetContainerSize(DataSize);
	m_FixedPts->Reserve(DataSize[0]);
		
	// Init container for set of moving points
	m_MovingPts = DatasetContainer::New();
	m_MovingPts->SetContainerSize(DataSize);
	m_MovingPts->Reserve(DataSize[0]);

	// Init flag for update
	m_GeometryUpdateRequired = true;
	m_ColorUpdateRequired = true;
	
	// Init container for intermediate transformation matrix
	m_TransformationMat = MatrixContainer::New();
	MatrixContainer::SizeType MatSize;
	MatSize.SetElement(0, 4*4);
	m_TransformationMat->SetContainerSize(MatSize);
	m_TransformationMat->Reserve(MatSize[0]);

	// Init standard deviation for noise
	m_NoiseStdDev = 0.f;

	// Init distribution type
	m_DistributionType = Uniform;

	// Init LUT coloring of points
	m_LUTEnabled = false;

	// Init transformation matrix (4x4 Identity matrix)
	float Tmp[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
	std::copy(Tmp, Tmp+16, stdext::checked_array_iterator<float*>(m_TransformationMat->GetBufferPointer(), 16));
}


//----------------------------------------------------------------------------
DataGenerator::~DataGenerator()
{
}


//----------------------------------------------------------------------------
void
DataGenerator::GenerateData()
{
	// Always return the same point cloud
	vtkMath::RandomSeed(0);

	// Update or generate spatial data
	if (m_GeometryUpdateRequired)
	{
		// Generate randomly distributed, but identical datasets; ranging from approx. [-128;127]
		// (distances then become similar to those in RGB cube [0;255])
		for (int i = 0; i < m_NumPts; ++i)
		{
			for (int j = 0; j < 3; ++j) // XYZ
			{
				float Val;
				if (m_DistributionType == Uniform)
				{
					switch (j)
					{
					case 0: Val = vtkMath::Random(-128., 127.); break; // X
					case 1: Val = vtkMath::Random(-128., 127.); break; // Y
					case 2: Val = vtkMath::Random(-128., 127.); break; // Z
					}
				}
				else if (m_DistributionType == Gaussian)
				{
					switch (j)
					{
					case 0: Val = vtkMath::Gaussian(0., 85.); break; // X
					case 1: Val = vtkMath::Gaussian(0., 10.); break; // Y
					case 2: Val = vtkMath::Gaussian(0., 50.); break; // Z
					}
				}

				m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+j] = Val;
				m_MovingPts->GetBufferPointer()[i*ICP_DATA_DIM+j] = Val;
			}
		}

		// Put noise on second data set if appropriate
		if (m_NoiseStdDev > FLT_EPSILON)
		{
			for (int i = 0; i < m_NumPts; ++i)
			{
				for (int j = 0; j < 3; ++j)
				{
					m_MovingPts->GetBufferPointer()[i*ICP_DATA_DIM+j] += vtkMath::Gaussian(0.f, m_NoiseStdDev);
				}
			}
		}
	}

	// Update or generate color information
	if (m_ColorUpdateRequired)
	{
		for (int i = 0; i < m_NumPts; ++i)
		{
			// RGB
			const float MAX_VAL = 255*3;
			if (m_LUTEnabled)
			{
				int LUTIdx = static_cast<int>(
					(
					m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+0] +
					m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+1] +
					m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+2] +
					128*3
					) / MAX_VAL * static_cast<float>(ritk::LUT_Jet_Length)
					);

				LUTIdx = std::min(LUTIdx, ritk::LUT_Jet_Length-1);

				m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+3] = static_cast<float>(ritk::LUT_Jet[LUTIdx][0])*255.f; 
				m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+4] = static_cast<float>(ritk::LUT_Jet[LUTIdx][1])*255.f;
				m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+5] = static_cast<float>(ritk::LUT_Jet[LUTIdx][2])*255.f;

				m_MovingPts->GetBufferPointer()[i*ICP_DATA_DIM+3] = static_cast<float>(ritk::LUT_Jet[LUTIdx][0])*255.f;
				m_MovingPts->GetBufferPointer()[i*ICP_DATA_DIM+4] = static_cast<float>(ritk::LUT_Jet[LUTIdx][1])*255.f;
				m_MovingPts->GetBufferPointer()[i*ICP_DATA_DIM+5] = static_cast<float>(ritk::LUT_Jet[LUTIdx][2])*255.f;
			}
			else
			{
				m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+3] = 255; 
				m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+4] = 0;
				m_FixedPts->GetBufferPointer()[i*ICP_DATA_DIM+5] = 0;

				m_MovingPts->GetBufferPointer()[i*ICP_DATA_DIM+3] = 0;
				m_MovingPts->GetBufferPointer()[i*ICP_DATA_DIM+4] = 200;
				m_MovingPts->GetBufferPointer()[i*ICP_DATA_DIM+5] = 0;
			}
		}
	}

	if (m_ColorUpdateRequired || m_GeometryUpdateRequired)
	{
		// Copy data to GPU
		m_FixedPts->SynchronizeDevice();
		m_MovingPts->SynchronizeDevice();
	}

	// Perform transformation on GPU (assuming that the transformation matrix is already on the device)
	CUDATransformPoints3D(
		m_MovingPts->GetCudaMemoryPointer(),
		m_TransformationMat->GetCudaMemoryPointer(),
		m_NumPts,
		ICP_DATA_DIM
		);

	// Update set of moving points on CPU
	m_MovingPts->SynchronizeHost();

	// Reset flags
	m_GeometryUpdateRequired = false;
	m_ColorUpdateRequired = false;
}


//----------------------------------------------------------------------------
void
DataGenerator::SetNumberOfPoints(unsigned long NumPts)
{
	if (m_NumPts != NumPts)
	{
		m_NumPts = NumPts;

		// Resize containers
		m_FixedPts = DatasetContainer::New();
		DatasetContainer::SizeType DataSize;
		DataSize.SetElement(0, ICP_DATA_DIM*m_NumPts);
		m_FixedPts->SetContainerSize(DataSize);
		m_FixedPts->Reserve(DataSize[0]);

		m_MovingPts = DatasetContainer::New();
		m_MovingPts->SetContainerSize(DataSize);
		m_MovingPts->Reserve(DataSize[0]);

		// Set flags
		m_ColorUpdateRequired = true;
		m_GeometryUpdateRequired = true;
	}
}


//----------------------------------------------------------------------------
void
DataGenerator::SetTransformationParameters(float RotX, float RotY, float RotZ, float TransX, float TransY, float TransZ)
{
	// Create the homogeneous matrix
	vtkSmartPointer<vtkTransform> Transform = vtkSmartPointer<vtkTransform>::New();
	Transform->RotateX(RotX);
	Transform->RotateY(RotY);
	Transform->RotateZ(RotZ);
	Transform->Translate(TransX, TransY, TransZ);
	Transform->Modified();

	// Copy to our matrix container
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			m_TransformationMat->GetBufferPointer()[r*4+c] = Transform->GetMatrix()->Element[r][c];

	// Synchronize to GPU
	m_TransformationMat->SynchronizeDevice();

	// Set flag
	m_GeometryUpdateRequired = true;
}