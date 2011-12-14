#ifndef REPGPU_H__
#define REPGPU_H__


//** @name	Representative structure used by RBC nearest neighbor search */
//@{
/// The struct
typedef struct
{
	/// Index of this representative among entire dataset
	unsigned long repIdx;

	/// Number of points in NN list of this representative
	unsigned long numPts;

	/// Pointer to indices of this representative's data points (in NN list)
	unsigned long* NNList;
} RepGPU;
//@}


#endif // REPGPU_H__