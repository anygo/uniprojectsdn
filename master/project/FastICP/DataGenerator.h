#ifndef DATAGENERATOR_H__
#define DATAGENERATOR_H__

#include "ritkCudaRegularMemoryImportImageContainer.h"


/**	@class		DataGenerator
 *	@brief		Class that generates synthetic data
 *	@author		Dominik Neumann
 *
 *	@details
 *	This is a class that generates two randomly distributed RGB-colored 3-D point sets that are both identical, except
 *	that the second point set can be transformed and made noisy.
 */
class DataGenerator
{
	/**	@name CUDA memory containers */
	//@{
	typedef ritk::CudaRegularMemoryImportImageContainerF DatasetContainer;
	typedef ritk::CudaRegularMemoryImportImageContainerF MatrixContainer;
	//@}

public:
	/// Constructor 
	DataGenerator();

	/// Destructor
	~DataGenerator();

	/// Generate synthetic data
	void GenerateData();

	/// Set rotation angles and translation vector
	void SetTransformationParameters(float RotX, float RotY, float RotZ, float TransX, float TransY, float TransZ);

	/// Set number of points per dataset
	void SetNumberOfPoints(unsigned long NumPts);

	/// Enum for supported distribution types
	enum DistributionType { Uniform, Gaussian };

	/// Set distribution type
	inline void SetDistributionType(DistributionType Type) { if (m_DistributionType != Type) { m_GeometryUpdateRequired = true; m_ColorUpdateRequired = true; } m_DistributionType = Type; }

	/// Toggle usage of LUT for coloring
	inline void SetColoring(bool LUTEnabled) { if (m_LUTEnabled != LUTEnabled) m_ColorUpdateRequired = true; m_LUTEnabled = LUTEnabled; }

	/// Set standard deviation of noise for second dataset
	inline void SetNoise(float StdDev) { if (m_NoiseStdDev != StdDev) m_GeometryUpdateRequired = true; m_NoiseStdDev = StdDev; }

	/// Returns pointer to the dataset container holding the fixed points
	inline DatasetContainer::Pointer GetFixedPtsContainer() const { return m_FixedPts; }

	/// Returns pointer to the dataset container holding the moving points
	inline DatasetContainer::Pointer GetMovingPtsContainer() const { return m_MovingPts; }

protected:
	/// Pointer to set of fixed points
	DatasetContainer::Pointer m_FixedPts;

	/// Pointer to set of moving points
	DatasetContainer::Pointer m_MovingPts;

	/// Container for transformation matrix
	MatrixContainer::Pointer m_TransformationMat;

	/// If this is set to true, the class will generate LUT-colored point sets
	bool m_LUTEnabled;

	/// Stores distribution type
	DistributionType m_DistributionType;

	/// Stores standard deviation for noise (applied to second point set)
	float m_NoiseStdDev;

	/// Stores number of points per dataset
	unsigned long m_NumPts;

	/// Indicates whether the spatial data has to be recomputed
	bool m_GeometryUpdateRequired;

	/// Indicates whether the color data has to be recomputed
	bool m_ColorUpdateRequired;

private:
	/// Purposely not implemented
	DataGenerator(DataGenerator&);

	/// Purposely not implemented
	void operator=(const DataGenerator&); 
};


#endif // DATAGENERATOR_H__