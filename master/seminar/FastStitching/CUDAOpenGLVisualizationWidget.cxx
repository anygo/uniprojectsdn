#include "CUDAOpenGLVisualizationWidget.h"

#include "OpenGLExtensions.h"
#include <cutil_inline.h>
#include <cuda_gl_interop.h>

#include <gl/gl.h>
#include <gl/glu.h>
#include "Shaders.h"
#include "LUTs.h"

#include <math.h>
#include <algorithm>

#include "Manager.h"
#include "CudaRegularMemoryImportImageContainer.h"


extern "C"
void
CUDARangeToWorld(float4* duplicate, const cudaArray *InputImageArray, float4 *DeviceOutput, int w, int h, float fx, float fy, float cx, float cy, float k1, float k2);




#define CHECK_GL_ERROR()													\
{																			\
	GLenum errCode = glGetError();											\
	if ( errCode != GL_NO_ERROR )											\
	{																		\
		std::cout << "An OpenGL error occured: " << errCode << std::endl;	\
		std::cout << gluErrorString(errCode) << std::endl;					\
		std::cout << __FILE__ << " Line " << __LINE__  << std::endl;		\
	}																		\
}


static void qNormalizeAngle(int &angle)
{
	while (angle < 0)
		angle += 360 * 16;
	while (angle > 360 * 16)
		angle -= 360 * 16;
}


CUDAOpenGLVisualizationWidget::CUDAOpenGLVisualizationWidget(QWidget *parent) :
QGLWidget(parent)
{
	//cutilSafeCall(cudaThreadExit());
	cutilSafeCall(cudaSetDevice(cutGetMaxGflopsDeviceId()));
	cutilSafeCall(cudaGLSetGLDevice(cutGetMaxGflopsDeviceId()));
	m_AllocatedSize = 0;
	m_InputImgArr = NULL;
	m_Output = NULL;

	m_CurWCs = NULL;
	m_PrevWCs = NULL;

	// Initialize GPU Memory to hold the previous and current world data
	cutilSafeCall(cudaMalloc((void**)&(m_CurWCs), 640*480*sizeof(float4)));
	cutilSafeCall(cudaMalloc((void**)&(m_PrevWCs), 640*480*sizeof(float4)));

	m_renderPoints = false;

	m_FullscreenFlag = false;

	// To accept key events
	setFocusPolicy(Qt::StrongFocus);

	// Register this smart pointer
	qRegisterMetaType<ritk::RImageF2::ConstPointer>("ritk::RImageF2::ConstPointer");

	// Not initialized by default
	m_InitFlag = false;

	// No data available
	m_CurrentFrame = NULL;

	// No texture so far
	m_TextureSize[0] = 0;
	m_TextureSize[1] = 0;
	m_TextureCoords = 0;
	m_RGBTextureData = 0;
	m_RangeTextureData = 0;

	m_ClippingPlanes[0] = 0;
	m_ClippingPlanes[1] = 0;

	// No translation by default
	m_Translation[0] = 0;
	m_Translation[1] = 0;

	// No rotation by default
	m_Rotation[0] = 0;
	m_Rotation[1] = 0;
	m_Rotation[2] = 0;

	// Std zoom
	m_RawZoom = 0.5f*log(0.5f/1.5f); // = atanh(y/2+1) 
	m_Zoom = (tanh(m_RawZoom)+1)*2;

	// VBOs not initialized by default
	m_VBOInitialized = false;

	m_Counters[0] = 0;
	m_Counters[1] = 0;
	m_Counters[2] = 0;
	m_Counters[3] = 0;
	m_Timers[0] = 0;
	m_Timers[1] = 1;
	m_Timers[2] = 1;
	m_Timers[3] = 1;

	// Shader
	m_LUTID = 0;
	m_Alpha = 0.5;
	m_ShaderProgram = 0;

	// Internal communication
	connect(this, SIGNAL(NewDataAvailable(bool)), this, SLOT(UpdateVBO(bool)), Qt::BlockingQueuedConnection);
	connect(this, SIGNAL(ResetCameraSignal()), this, SLOT(ResetCamera()));
}


//----------------------------------------------------------------------------
CUDAOpenGLVisualizationWidget::~CUDAOpenGLVisualizationWidget()
{
	// Clean up
	ritk::glDeleteBuffers(1, &m_VBOVertices);

	// Free GPU Memory that holds the previous and current world data
	cutilSafeCall(cudaFree(m_CurWCs));
	cutilSafeCall(cudaFree(m_PrevWCs));

	if ( m_TextureCoords )
		delete[] m_TextureCoords;

	if ( m_RGBTextureData )
		delete[] m_RGBTextureData;

	if ( m_RangeTextureData )
		delete[] m_RangeTextureData;

	if(m_InputImgArr)
		cutilSafeCall(cudaFreeArray(m_InputImgArr));
}

//----------------------------------------------------------------------------
void CUDAOpenGLVisualizationWidget::Stitch() 
{

	
}

//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::SetRangeData(ritk::RImageF2::ConstPointer Data)
{
	// Lock ourself
	m_Mutex.lock();

	// Initialize timer
	LONGLONG Frequency;
	QueryPerformanceFrequency((LARGE_INTEGER*)&Frequency);

	LONGLONG C1;
	QueryPerformanceCounter((LARGE_INTEGER*)&C1);

	bool ResetCameraRequired = false;
	if ( !m_CurrentFrame )
	{
		ResetCameraRequired = true;
	}

	// Update the current frame
	m_CurrentFrame = Data;

	// For convenience
	long SizeX = m_CurrentFrame->GetBufferedRegion().GetSize()[0];
	long SizeY = m_CurrentFrame->GetBufferedRegion().GetSize()[1];

	// Flag that indicates whether the texture size changed
	bool SizeChanged = SizeX != m_TextureSize[0] || SizeY != m_TextureSize[1];

	// Prepare the texture coords
	if (SizeChanged)
	{
		// Delete old memory and allocate memory to fit the requirements
		if ( m_TextureCoords )
			delete[] m_TextureCoords;

		m_TextureCoords = new GLfloat[SizeX*SizeY*2 + SizeX*(SizeY-2)*2];

		// Update coords
		for ( long l = 0; l < SizeX*SizeY + SizeX*(SizeY-2); l++ )
		{
			if (l%2==0)
			{
				m_TextureCoords[l*2+0] = (l%(SizeX*2))/((SizeX*2)*1.f);
				m_TextureCoords[l*2+1] = (l/(SizeX*2))/(SizeY*1.f) ;
			}
			else
			{
				m_TextureCoords[l*2+0] = (l%(SizeX*2)-1)/((SizeX*2)*1.f);
				m_TextureCoords[l*2+1] = (l/(SizeX*2)+1)/(SizeY*1.f);
			}
		}

	}

	// Prepare the rgb texture data
	if (SizeChanged)
	{
		// Delete old memory and allocate memory to fit the requirements
		if ( m_RGBTextureData )
			delete[] m_RGBTextureData;
		m_RGBTextureData = new unsigned char[SizeX*SizeY*3];
	}

	// Copy the RGB texture
	const ritk::RImageF2::RGBType *RGBData = m_CurrentFrame->GetRGBImage()->GetBufferPointer();
	unsigned char *RGBTexturePtr = m_RGBTextureData;
	for ( int l = 0; l < SizeX*SizeY; ++l, ++RGBTexturePtr, ++RGBData )
	{
		*RGBTexturePtr = (unsigned char)(*RGBData)[0];
		*(++RGBTexturePtr) = (unsigned char)(*RGBData)[1];
		*(++RGBTexturePtr) = (unsigned char)(*RGBData)[2];
	}	

	// Prepare the range texture data
	if (SizeChanged)
	{
		// Delete old memory and allocate memory to fit the requirements
		if ( m_RangeTextureData )
			delete[] m_RangeTextureData;
		m_RangeTextureData = new unsigned char[SizeX*SizeY];
	}

	// Clamp and rescale the range information
	unsigned char *RTexturePtr = m_RangeTextureData;
	const float *RData = m_CurrentFrame->GetRangeImage()->GetBufferPointer();
	float Scale = 1.f/(m_RangeBoundaries[1]-m_RangeBoundaries[0]);

	for ( long l = 0; l < SizeX*SizeY; l++, RTexturePtr++ )
	{
		float val = RData[l];
		if ( val < m_RangeBoundaries[0] )
			val = m_RangeBoundaries[0];
		if ( val > m_RangeBoundaries[1] )
			val = m_RangeBoundaries[1];
		*RTexturePtr = (unsigned char)(((val - m_RangeBoundaries[0])*Scale)*255);
	}

	// Remember the texture size
	m_TextureSize[0] = SizeX;
	m_TextureSize[1] = SizeY;


	// Unlock ourself
	m_Mutex.unlock();

	// Delegate the data to the UpdateVBO slot that is connected to the NewDataAvailable signal.
	emit NewDataAvailable(SizeChanged);

	// Reset the camera
	if ( ResetCameraRequired )
		emit ResetCameraSignal();
}


//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::UpdateVBO(bool SizeChanged)
{
	// Lock the mutex
	m_Mutex.lock();

	// To be on the safe side
	this->makeCurrent();

	// The number of vertices that we have to process
	/*long SizeX = m_CurrentFrame->GetBufferedRegion().GetSize()[0];
	long SizeY = m_CurrentFrame->GetBufferedRegion().GetSize()[1];*/
	long SizeX = m_TextureSize[0];
	long SizeY = m_TextureSize[1];

	if(SizeX * SizeY != m_AllocatedSize)
	{
		if(m_InputImgArr)
			cutilSafeCall(cudaFreeArray(m_InputImgArr));

		cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc(32,0,0,0,cudaChannelFormatKindFloat);
		cutilSafeCall(cudaMallocArray(&m_InputImgArr,&ChannelDesc,SizeX,SizeY));

		cutilSafeCall(cudaGraphicsUnregisterResource(m_Cuda_vbo_resource));
		ritk::glBindBuffer(GL_ARRAY_BUFFER, m_VBOVertices);
		ritk::glBufferData(GL_ARRAY_BUFFER, SizeX*SizeY * 4*sizeof(float) + SizeX*(SizeY-2) * 4*sizeof(float), 0, GL_DYNAMIC_DRAW);
		cutilSafeCall(cudaGraphicsGLRegisterBuffer(&m_Cuda_vbo_resource, m_VBOVertices, cudaGraphicsMapFlagsWriteDiscard));

		m_AllocatedSize = SizeX * SizeY;
	}

	// Copy the input data to the device
	cutilSafeCall(cudaMemcpyToArray(m_InputImgArr, 0, 0, m_CurrentFrame->GetRangeImage()->GetBufferPointer(), SizeX*SizeY * sizeof(float), cudaMemcpyHostToDevice));

	// CUDA/OpenGL interoperability
	size_t num_bytes;
	cutilSafeCall(cudaGraphicsMapResources(1, &m_Cuda_vbo_resource, 0));
	cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&m_Output, &num_bytes, m_Cuda_vbo_resource));


	float4* tmp = m_CurWCs;
	m_CurWCs = m_PrevWCs;
	m_PrevWCs = tmp;

	// Compute the world coordinates
	CUDARangeToWorld
		(
		m_CurWCs,
		m_InputImgArr, 
		m_Output,
		m_CurrentFrame->GetBufferedRegion().GetSize()[0],
		m_CurrentFrame->GetBufferedRegion().GetSize()[1],
		m_CurrentFrame->GetCameraParameters()[ritk::CameraParameters::FX],
		m_CurrentFrame->GetCameraParameters()[ritk::CameraParameters::FY],
		m_CurrentFrame->GetCameraParameters()[ritk::CameraParameters::CX],
		m_CurrentFrame->GetCameraParameters()[ritk::CameraParameters::CY],
		m_CurrentFrame->GetCameraParameters()[ritk::CameraParameters::K1],
		m_CurrentFrame->GetCameraParameters()[ritk::CameraParameters::K2]
		);

	// Release CUDA resources
	cutilSafeCall(cudaGraphicsUnmapResources(1, &m_Cuda_vbo_resource, 0));

	CHECK_GL_ERROR();

	LONGLONG C2;
	QueryPerformanceCounter((LARGE_INTEGER*)&C2);

	// Activate the texture coordinate VBO and update its data
	if (SizeChanged)
	{
		ritk::glBindBuffer(GL_ARRAY_BUFFER, m_VBOTexCoords);
		ritk::glBufferData(GL_ARRAY_BUFFER, SizeX*SizeY * 2 * sizeof(float) + SizeX*(SizeY-2)*2*sizeof(float), m_TextureCoords, GL_DYNAMIC_DRAW);
	}
	CHECK_GL_ERROR();

	// Bind the RGB texture
	glBindTexture(GL_TEXTURE_2D, m_RGBTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SizeX, SizeY, 0, GL_RGB, GL_UNSIGNED_BYTE, m_RGBTextureData);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	CHECK_GL_ERROR();

	// Bind the range texture
	glBindTexture(GL_TEXTURE_2D, m_RangeTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, SizeX, SizeY, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, m_RangeTextureData);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	CHECK_GL_ERROR();

	LONGLONG C3;
	QueryPerformanceCounter((LARGE_INTEGER*)&C3);

	// Synchronize
	m_VBOInitialized = true;

	// Unlock
	m_Mutex.unlock();

	// Update the OpenGL state
	updateGL();

	// Wait for all OpenGL operations to finish
	glFinish();
}


//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::ResetCamera()
{
	// Lock the mutex
	m_Mutex.lock();

	// Sanity check
	if ( !m_CurrentFrame )
	{
		std::cerr << "Could not reset camera: no frame provided!" << std::endl;
		m_Mutex.unlock();
		return;
	}

	// Check for previous OpenGL errors
	CHECK_GL_ERROR();

	int addTmp = 3;
	long NumVertices = m_CurrentFrame->GetWorldCoordImage()->GetRequestedRegion().GetNumberOfPixels();
	float *CoordPtr = (float*)m_CurrentFrame->GetWorldCoordImage()->GetBufferPointer();
	// Compute the bounding sphere of the scene
	// The number of vertices that we have to process

	float *realOut;
	if(m_Output)
	{
		long SizeX = m_CurrentFrame->GetBufferedRegion().GetSize()[0];
		long SizeY = m_CurrentFrame->GetBufferedRegion().GetSize()[1];

		// for triangle-view in opengl we need most of the vertices twice to get a surface
		NumVertices = SizeX*SizeY + SizeX*(SizeY-2);
		addTmp = 4;

		realOut = new float[NumVertices*4];

		size_t num_bytes;

		cutilSafeCall(cudaGraphicsMapResources(1, &m_Cuda_vbo_resource, 0));
		cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&m_Output, &num_bytes, m_Cuda_vbo_resource));
		cutilSafeCall(cudaMemcpy(realOut, m_Output, num_bytes, cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaGraphicsUnmapResources(1, &m_Cuda_vbo_resource, 0));

		CoordPtr = realOut;
	}

	float BBMin[] = {1e16,1e16,1e16};
	float BBMax[] = {-1e16,-1e16,-1e16};
	float Center[] = {0,0,0};
	for ( long l = 0; l < NumVertices; l++, CoordPtr += addTmp )
	{
		if ( fabsf(*(CoordPtr+2)) < 1e-6 )
			continue;
		if ( *CoordPtr < BBMin[0] )
			BBMin[0] = *CoordPtr;
		if ( *(CoordPtr+1) < BBMin[1] )
			BBMin[1] = *(CoordPtr+1);
		if ( *(CoordPtr+2) < BBMin[2] )
			BBMin[2] = *(CoordPtr+2);

		if ( *CoordPtr > BBMax[0] )
			BBMax[0] = *CoordPtr;
		if ( *(CoordPtr+1) > BBMax[1] )
			BBMax[1] = *(CoordPtr+1);
		if ( *(CoordPtr+2) > BBMax[2] )
			BBMax[2] = *(CoordPtr+2);
	}
	Center[0] = (BBMin[0] + BBMax[0])/2.f;
	Center[1] = (BBMin[1] + BBMax[1])/2.f;
	Center[2] = (BBMin[2] + BBMax[2])/2.f;
	float w1 = BBMax[0] - BBMin[0];
	float w2 = BBMax[1] - BBMin[1];
	float w3 = BBMax[2] - BBMin[2];
	w1 *= w1;
	w2 *= w2;
	w3 *= w3;
	float Radius = w1 + w2 + w3;
	Radius = ( Radius == 0 ) ? (1.f) : (2*Radius);
	Radius = sqrt(Radius)*0.5f;

	// Set up our viewing direction
	m_EyePos[0] = 0;
	m_EyePos[1] = 0;
	m_EyePos[2] = 0;
	m_ViewCenter[0] = Center[0];
	m_ViewCenter[1] = Center[1];
	m_ViewCenter[2] = Center[2];

	// Update clipping planes
	m_ClippingPlanes[0] = 1;
	m_ClippingPlanes[1] = m_ClippingPlanes[0] + 2.1f*Radius;

	// Set up viewing frustum 
	m_Zoom = 1.0;
	m_RawZoom = 0.5f*log(0.5f/1.5f);
	glMatrixMode(GL_LINEAR);
	glLoadIdentity();
	gluPerspective((GLfloat)45.0f*m_Zoom, (GLfloat)(m_Width)/(GLfloat)(m_Height), (GLfloat)m_ClippingPlanes[0], m_ClippingPlanes[1]);

	// Reset translation
	m_Translation[0] = 0;
	m_Translation[1] = 0;

	// Reset rotation
	m_Rotation[0] = 0;
	m_Rotation[1] = 0;
	m_Rotation[2] = 0;

	// Unlock the mutex
	m_Mutex.unlock();

	if(m_Output)
		delete[] realOut;
}


//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::initializeGL()
{
	// Get the OpenGL extension function pointers...
	if ( !ritk::InitOpenGLExtensions() )
	{
		std::cerr << "Could not initialize OpenGL extensions!" << std::endl;
		CHECK_GL_ERROR();
		return;
	}

	// Create the VBOs
	ritk::glGenBuffers(1, &m_VBOVertices);
	ritk::glBindBuffer(GL_ARRAY_BUFFER, m_VBOVertices);

	// Just a dummy size to prevent some errors if no memory is allocated
	ritk::glBufferData(GL_ARRAY_BUFFER, 4, 0, GL_DYNAMIC_DRAW);


	cutilSafeCall(cudaGraphicsGLRegisterBuffer(&m_Cuda_vbo_resource, m_VBOVertices, cudaGraphicsMapFlagsWriteDiscard));

	ritk::glGenBuffers(1, &m_VBOTexCoords);
	ritk::glBindBuffer(GL_ARRAY_BUFFER, m_VBOTexCoords);

	CHECK_GL_ERROR();

	// Create textures
	glGenTextures(1, &m_RGBTexture);
	glGenTextures(1, &m_LUTTexture);
	glGenTextures(1, &m_RangeTexture);

	CHECK_GL_ERROR();

	// Shaders
	LoadShader(ritk::VertexShader_LUT, true);
	LoadShader(ritk::FragmentShader_LUT, false);
	ritk::glLinkProgram(m_ShaderProgram);

	CHECK_GL_ERROR();

	// Std OpenGL
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glEnable(GL_DEPTH_TEST);

	CHECK_GL_ERROR();

	// We're initialized right now. Update the flag
	m_InitFlag = true;
}


//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::resizeGL(int width, int height)
{
	m_Mutex.lock();

	// Reset The Current Viewport
	glViewport(0, 0, (GLsizei)(width), (GLsizei)(height));

	// Update members
	m_Width = width;
	m_Height = height;

	// The projection matrix
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective((GLfloat)45.0f*m_Zoom, (GLfloat)(m_Width)/(GLfloat)(m_Height), (GLfloat)m_ClippingPlanes[0], m_ClippingPlanes[1]);

	// Select the modelview matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	CHECK_GL_ERROR();

	m_Mutex.unlock();
}


//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::paintGL()
{
	// Lock
	m_Mutex.lock();

	// Check for previous OpenGL errors
	CHECK_GL_ERROR();

	// If we have no data, return
	if ( !m_CurrentFrame )
	{
		m_Mutex.unlock();
		return;
	}

	// If the current data and the VBO are not synchronized return
	if ( !m_VBOInitialized )
	{
		m_Mutex.unlock();
		return;
	}

	// Set up viewing frustum defined by our bounding box
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective((GLfloat)45.0f*m_Zoom, (GLfloat)(m_Width)/(GLfloat)(m_Height), (GLfloat)m_ClippingPlanes[0], (GLfloat)m_ClippingPlanes[1]);

	// Std OpenGL
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Set up our viewing direction
	gluLookAt(m_EyePos[0], m_EyePos[1], m_EyePos[2], 
		m_ViewCenter[0], m_ViewCenter[1], m_ViewCenter[2], 
		0, 1, 0);

	// Apply the current translation
	glTranslatef(m_Translation[0],m_Translation[1],0);

	// Apply the current rotation
	glTranslatef(m_ViewCenter[0],m_ViewCenter[1],m_ViewCenter[2]);
	glRotatef(m_Rotation[0] / 16.0, 1.0, 0.0, 0.0);
	glRotatef(m_Rotation[1] / 16.0, 0.0, 1.0, 0.0);
	glRotatef(m_Rotation[2] / 16.0, 0.0, 0.0, 1.0);
	glTranslatef(-m_ViewCenter[0],-m_ViewCenter[1],-m_ViewCenter[2]);

	// Default drawing color
	glColor3f(1,1,1);

	// Enable shader program
	ritk::glUseProgram(m_ShaderProgram);

	// LUT texture
	BindLUT(m_LUTID);
	GLint LUTTextureLocation = ritk::glGetUniformLocation(m_ShaderProgram, "m_LUTTexture");
	ritk::glActiveTexture(GL_TEXTURE0);

	glBindTexture(GL_TEXTURE_1D, m_LUTTexture);
	glEnable(GL_TEXTURE_1D);
	ritk::glUniform1i(LUTTextureLocation, 0);

	// The current alpha value for blending
	GLint AlphaLocation = ritk::glGetUniformLocation(m_ShaderProgram, "m_Alpha");
	ritk::glUniform1f(AlphaLocation, m_Alpha);

	// The RGB image texture
	GLint RGBTextureLocation = ritk::glGetUniformLocation(m_ShaderProgram, "m_RGBTexture");
	ritk::glActiveTexture(GL_TEXTURE1); 
	glBindTexture(GL_TEXTURE_2D, m_RGBTexture);
	glEnable(GL_TEXTURE_2D);
	ritk::glUniform1i(RGBTextureLocation, 1);

	// The range image texture
	GLint RangeTextureLocation = ritk::glGetUniformLocation(m_ShaderProgram, "m_RangeTexture");
	ritk::glActiveTexture(GL_TEXTURE2); 

	glBindTexture(GL_TEXTURE_2D, m_RangeTexture);
	glEnable(GL_TEXTURE_2D);
	ritk::glUniform1i(RangeTextureLocation, 2);

	// Render the point cloud using the VBOs
	glEnableClientState(GL_VERTEX_ARRAY);
	ritk::glBindBuffer(GL_ARRAY_BUFFER, m_VBOVertices);

	glVertexPointer(4, GL_FLOAT, 0, NULL);

	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	ritk::glBindBuffer(GL_ARRAY_BUFFER, m_VBOTexCoords);

	glTexCoordPointer(2, GL_FLOAT, 0, NULL);

	int outputWidth = m_CurrentFrame->GetBufferedRegion().GetSize()[0]*2;
	for(int i=0; i<m_CurrentFrame->GetBufferedRegion().GetSize()[1]-1; ++i)
	{
		if(m_renderPoints)
			glDrawArrays(GL_POINTS, i*outputWidth, outputWidth);
		else
			glDrawArrays(GL_TRIANGLE_STRIP, i*outputWidth, outputWidth);
	}

	// Check for GL errors
	CHECK_GL_ERROR();

	// Don't forget to unlock
	m_Mutex.unlock();
}



//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::SetTranslation(int dx, int dy)
{
	m_Translation[0] -= dx;
	m_Translation[1] -= dy;
	/*m_EyePos[0] += dx;
	m_EyePos[1] += dy;
	m_ViewCenter[0] += dx;
	m_ViewCenter[1] += dy;*/
}


//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::SetXRotation(int angle)
{
	qNormalizeAngle(angle);
	if ( angle != m_Rotation[0] ) 
	{
		m_Rotation[0] = angle;
	}
}


//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::SetYRotation(int angle)
{
	qNormalizeAngle(angle);
	if ( angle != m_Rotation[1] ) 
	{
		m_Rotation[1] = angle;
	}
}


//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::SetZRotation(int angle)
{
	qNormalizeAngle(angle);
	if ( angle != m_Rotation[2] ) 
	{
		m_Rotation[2] = angle;
	}
}


//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::UpdateZoom(float delta)
{
	// Lock the mutex
	m_Mutex.lock();

	// Update the zoom
	m_RawZoom += delta;
	m_Zoom = (tanh(m_RawZoom)+1)*2;

	// Lock the mutex
	m_Mutex.unlock();
}


//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::SetRangeClamping(float min, float max)
{
	m_Mutex.lock();

	m_RangeBoundaries[0] = min;
	m_RangeBoundaries[1] = max;

	m_Mutex.unlock();
}


//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::SetLUTAlpha(float value)
{
	m_Mutex.lock();

	m_Alpha = value;

	m_Mutex.unlock();

	updateGL();
}


//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::SetLUT(unsigned int ID)
{
	m_Mutex.lock();

	m_LUTID = ID;

	m_Mutex.unlock();

	updateGL();
}


//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::SetRenderType(bool renderPoints)
{
	m_renderPoints = renderPoints;
}


//----------------------------------------------------------------------------
bool
CUDAOpenGLVisualizationWidget::BindLUT(unsigned int LUTID)
{
	CHECK_GL_ERROR();

	// Get LUT and normalize
	unsigned int NumLUTElems = 0;
	GLfloat *LUT = 0;
	if ( LUTID == 0 )
		LUT = ritk::GetLUT("FireStorm1", &NumLUTElems);

	else if ( LUTID == 1 )
		LUT = ritk::GetLUT("FireStorm2", &NumLUTElems);

	else if ( LUTID == 2 )
		LUT = ritk::GetLUT("ColdFusion", &NumLUTElems);

	else if ( LUTID == 3 )
		LUT = ritk::GetLUT("HylaArborea", &NumLUTElems);

	else if ( LUTID == 4 )
		LUT = ritk::GetLUT("Jet", &NumLUTElems);

	else
	{
		std::cerr << "Invalid LUTID!" << std::endl;
		m_Mutex.unlock();
		return false;
	}

	glBindTexture(GL_TEXTURE_1D, m_LUTTexture);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	CHECK_GL_ERROR();
	//glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, NumLUTElems, 0, GL_RGBA, GL_FLOAT, LUT);
	CHECK_GL_ERROR();
	glBindTexture(GL_TEXTURE_1D, 0);
	CHECK_GL_ERROR();

	GLenum errCode = glGetError();
	CHECK_GL_ERROR();

	return true;
}


//----------------------------------------------------------------------------
GLuint
CUDAOpenGLVisualizationWidget::LoadShader(const char *Program, bool VertexFlag)
{
	// No valid program
	if ( !Program )
	{
		std::cerr << "No valid program provided!" << std::endl;
		return 0;
	}

	// Create shader. The type is set by the user
	GLuint ShaderHandle = 0;
	if ( VertexFlag )
		ShaderHandle = ritk::glCreateShader(GL_VERTEX_SHADER);

	else
		ShaderHandle = ritk::glCreateShader(GL_FRAGMENT_SHADER);

	// Set source
	ritk::glShaderSource(ShaderHandle, 1, &Program, NULL);

	// Compile shader
	ritk::glCompileShader(ShaderHandle);

	// If the shader program does not exist, create it
	if ( !m_ShaderProgram )
		m_ShaderProgram = ritk::glCreateProgram();

	// Finally attach the shader to the shader program and save the handle
	ritk::glAttachShader(m_ShaderProgram,ShaderHandle);

	m_ShaderHandles.push_back(ShaderHandle);

	// Some debug info
	int infologLength = 0;
	int charsWritten  = 0;
	char *infoLog;
	ritk::glGetShaderiv(ShaderHandle, GL_INFO_LOG_LENGTH, &infologLength);

	if (infologLength > 0)
	{
		infoLog = (char *)malloc(infologLength);
		ritk::glGetShaderInfoLog(ShaderHandle, infologLength, &charsWritten, infoLog);

		free(infoLog);
	}
	return ShaderHandle;
}


//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::mousePressEvent(QMouseEvent *e)
{
	lastPos = e->pos();
}


//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::mouseMoveEvent(QMouseEvent *e)
{
	int dx = e->x() - lastPos.x();
	int dy = e->y() - lastPos.y();

	if ( e->buttons() & Qt::LeftButton ) 
	{
		SetXRotation(m_Rotation[0] - 4 * dy);
		SetYRotation(m_Rotation[1] + 4 * dx);
	} 
	else if ( e->buttons() & Qt::RightButton ) 
	{
		SetXRotation(m_Rotation[0] - 4 * dy);
		SetZRotation(m_Rotation[2] + 4 * dx);
	}
	else if (e->buttons() & Qt::MidButton )
	{
		SetTranslation(3 * dx, 3 * dy);
	}
	lastPos = e->pos();

	updateGL();
}


//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::wheelEvent(QWheelEvent *e)
{
	UpdateZoom(-e->delta()/2000.f);
	e->accept();

	updateGL();
}


//----------------------------------------------------------------------------
void 
CUDAOpenGLVisualizationWidget::keyPressEvent(QKeyEvent *e)
{
	if ( e->key() == Qt::Key_R )
	{
		ResetCamera();
		e->accept();
	}

	if ( e->key() == Qt::Key_F1 )
	{
		m_FullscreenFlag = !m_FullscreenFlag;
		SetFullScreenMode(m_FullscreenFlag);
	}

	updateGL();	
}

//----------------------------------------------------------------------------
void
CUDAOpenGLVisualizationWidget::SetFullScreenMode(bool b)
{
	if ( b )
	{
		std::cout << "Going fullscreen..." << std::endl;

		// Make our window without panels
		this->setWindowFlags( Qt::FramelessWindowHint | Qt::Tool | Qt::WindowStaysOnTopHint );

		// Resize refer to desktop
		this->setFocus();
		this->showFullScreen();

		this->setWindowState(Qt::WindowFullScreen);
	}
	else
	{
		std::cout << "Going normal..." << std::endl;
		this->setWindowFlags(0);
		this->showNormal();
	}
}

