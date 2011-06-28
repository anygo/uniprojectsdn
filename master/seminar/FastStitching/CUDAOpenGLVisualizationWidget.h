#ifndef CUDAOPENGLVISUALIZATIONWIDGET_H__
#define CUDAOPENGLVISUALIZATIONWIDGET_H__

#include <QtOpenGL>
#include <QGLWidget>

#include <QMutex>

#include "RImage.h"

#include "CudaContext.h"

#include <vtkSmartPointer.h>
#include <vtkMatrix4x4.h>


/**
*	@class		CUDAOpenGLVisualizationWidget
*	@author		Dominik Neumann and Felix Lugauer
*	@brief		
*
*	@details
*/
class CUDAOpenGLVisualizationWidget : public QGLWidget
{
	Q_OBJECT

	public slots:
		void Stitch();

public:

	/// Constructor
	CUDAOpenGLVisualizationWidget(QWidget *parent=0);
	/// Destructor
	~CUDAOpenGLVisualizationWidget();

	void Prepare(ritk::RImageF2::ConstPointer Data);

	

signals:
	void NewDataAvailable(bool SizeChanged);
	void FrameStitched(float4*);

	protected slots:
		void UpdateVBO(bool SizeChanged);


protected:

	// Width and height of the drawing area
	int m_Width;
	int m_Height;

	/// Current near and farplane
	float m_ClippingPlanes[2];

	/// Current eye position and view center
	//@{
	float m_EyePos[3];
	float m_ViewCenter[3];
	//@}

	/// The current rotation
	int m_Rotation[3];
	/// The current translation
	int m_Translation[2];
	/// Last mouse point
	QPoint lastPos;
	/// The raw zoom value
	float m_RawZoom;
	/// The current zoom
	float m_Zoom;

	/// Mutex used to synchronize rendering and data update
	QMutex m_Mutex;

	/// The current frame
	ritk::RImageF2::ConstPointer m_CurrentFrame;
	// to check if the allocated size has changed
	size_t m_AllocatedSize;
	// for the input data
	cudaArray* m_InputImgArr;
	// for registerung the opengl resources to cuda
	cudaGraphicsResource* m_Cuda_vbo_resource;
	// output pointer for the GPU
	float4* m_Output;

	/// Range clamping
	float m_RangeBoundaries[2];


	/// The texture
	//@{
	int m_TextureSize[2];
	GLfloat *m_TextureCoords;
	unsigned char *m_RGBTextureData;
	unsigned char *m_RangeTextureData;
	//@}

	/// The VBO for the vertex coords
	GLuint m_VBOVertices;
	/// The VBO for the texture coords
	GLuint m_VBOTexCoords;


	/// The RGB texture
	GLuint m_RGBTexture;
	/// The range data texture
	GLuint m_RangeTexture;
	/// The LUT texture
	GLuint m_LUTTexture;

	/// The shader program
	GLuint m_ShaderProgram;
	/// Handles to the shader objects
	std::vector<GLuint> m_ShaderHandles;

	/// The current LUT index
	unsigned int m_LUTID;

	/// The current alpha value for blending
	float m_Alpha;

	/// Performance counters
	int m_Counters[10];
	float m_Timers[10];

	/// true if points instead of triangles
	bool m_renderPoints;


	float4* m_CurWCs;
	float4* m_PrevWCs;
	vtkSmartPointer<vtkMatrix4x4> m_PrevTrans;
};

#endif // CUDAOPENGLVISUALIZATIONWIDGET_H__