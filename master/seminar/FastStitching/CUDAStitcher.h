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
*	@class		CUDAStitcher
*	@author		Dominik Neumann and Felix Lugauer
*	@brief		
*
*	@details
*/
class CUDAStitcher : public QGLWidget
{
	Q_OBJECT

	public slots:
		void Stitch();

public:

	/// Constructor
	CUDAStitcher(QWidget *parent=0);
	/// Destructor
	~CUDAStitcher();

	void Prepare(ritk::RImageF2::ConstPointer Data);

	

signals:
	void Prepared(bool SizeChanged);
	void FrameStitched(float4*);

	protected slots:
		void UpdateVBO(bool SizeChanged);


protected:



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

	/// Handles to the shader objects
	std::vector<GLuint> m_ShaderHandles;


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