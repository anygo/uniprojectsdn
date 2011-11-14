#ifndef RBCOneShot_H__
#define RBCOneShot_H__

/**	@class		RBCOneShot
 *	@brief		efficient NN search
 *	@author		Dominik Neumann
 *
 *	@details
 *	Class for efficient NN search using the one-shot Random Ball Cover (RBC) data structure,
 *	CUDA required.
 */
class RBC {
	
	// representative on CPU
	typedef struct Rep
	{
		unsigned int index;
		std::list<unsigned int> points;
	} Rep;
	
	// representative on GPU
	typedef struct RepGPU
	{
		uint index;
		uint pts;
		uint* dev_points;
	} RepGPU;

public:
	RBC(float** dataset, uint dim, uint pts);
	RBC(float** dataset, uint dim, uint pts, uint reps);
	~RBC();
	
	uint* query(float**);
	
private:
	void initGPUMem();
	void releaseGPUMem();
	
	void buildRBC();
	
	// inidcates a required rebuild of the RBC data structure (e.g. due to updated dataset)
	bool rebuildRequired = true;

	uint m_Dim;
	uint m_Pts;
}

#endif // RBCOneShot_H__