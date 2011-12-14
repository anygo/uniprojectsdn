#ifndef DEFS_H__
#define DEFS_H__

// In this plugin, we always deal with 6-D data, XYZ + RGB
#define ICP_DATA_DIM 6

// Kinect data definitions
#define KINECT_IMAGE_WIDTH 640
#define KINECT_IMAGE_HEIGHT 480

// Maximum number of representatives
#ifndef MAX_REPS
#define MAX_REPS 256
#endif

// # Threads per block
#ifndef CUDA_THREADS_PER_BLOCK
#define CUDA_THREADS_PER_BLOCK 128
#endif

// Buffer size used for allocating shared memory
#ifndef CUDA_BUFFER_SIZE
#define CUDA_BUFFER_SIZE 128
#endif

// Used in .cu-files
#define DivUp(a,b) ((a % b != 0) ? (a/b + 1) : (a/b))

// Macro
//#define __COMPUTERUNTIME(CODE,NAME) CODE
#define __COMPUTERUNTIME(CODE,NAME)													\
{																					\
	QTime TIMER; const int NTIMES = 250; int ELAPSED;								\
	TIMER.start();																	\
	for (int i = 0; i < NTIMES; ++i) { CODE }										\
	ELAPSED = TIMER.elapsed();														\
	std::cout << NAME << ": " << (float)ELAPSED/(float)NTIMES << "ms" << std::endl; \
}

#endif // DEFS_H