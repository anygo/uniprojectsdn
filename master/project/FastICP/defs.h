#ifndef DEFS_H__
#define DEFS_H__

// In this plugin, we always deal with 6-D data, XYZ + RGB
#ifndef ICP_DATA_DIM
#define ICP_DATA_DIM 6
#endif

// Kinect data definitions
#ifndef KINECT_IMAGE_WIDTH
#define KINECT_IMAGE_WIDTH 640
#endif
#ifndef KINECT_IMAGE_HEIGHT
#define KINECT_IMAGE_HEIGHT 480
#endif

// Maximum number of representatives
#ifndef MAX_REPS
#define MAX_REPS 256
#endif

// # Threads per block (do NOT change this value: 128)
#ifndef CUDA_THREADS_PER_BLOCK
#define CUDA_THREADS_PER_BLOCK 128
#endif

// Buffer size used for allocating shared memory
#ifndef CUDA_BUFFER_SIZE
#define CUDA_BUFFER_SIZE CUDA_THREADS_PER_BLOCK
#endif

// Use texture memory instead of global memory in some situations (buggy!)
//#define USE_TEXTURE_MEMORY

// Used in .cu-files
#ifndef DIVUP
#define DIVUP(a,b) ((a % b != 0) ? (a/b + 1) : (a/b))
#endif


#endif // DEFS_H