#ifndef CUDAOPENGLVISUALIZATIONWIDGET_H__
#define CUDAOPENGLVISUALIZATIONWIDGET_H__

#include <QtOpenGL>
#include <QGLWidget>

#include <QMutex>

#include "RImage.h"

#include "CudaContext.h"

/**
*	@class		CUDAOpenGLVisualizationWidget
*	@author		Jakob Wasza
*	@brief		A visualization widget that uses OpenGL
*
*	@details
*/
class CUDAOpenGLVisualizationWidget : public QGLWidget
{
	Q_OBJECT

public:

	/// Constructor
	CUDAOpenGLVisualizationWidget(QWidget *parent=0);
	/// Destructor
	~CUDAOpenGLVisualizationWidget();

	/**	@brief	Set the range data to render 
	*	@param	Data	The data to render
	*
	*	@details
	*	A call to this method will set the range data to render.
	*	In order to comply with the OpenGL render context this method
	*	will pass the data to the UpdateVBO() slot via the NewDataAvailable() signal.
	*/
	void SetRangeData(ritk::RImageF2::ConstPointer Data);

	/// Set the alpha value for blending
	void SetLUTAlpha(float value);

	/// Set the LUT ID
	void SetLUT(unsigned int ID);

	/// Set the render type of the data
	void SetRenderType(bool renderPoints);

	// Set the range clamping
	void SetRangeClamping(float min, float max);

	void Stitch();

signals:
	/// Signal to establish communication between SetTOFData and NewDataAvailable
	void NewDataAvailable(bool SizeChanged);
	/// Signal to establish communication with the ResetCamera method
	void ResetCameraSignal();


protected:
	/**	@name	Does the OpenGL part */
	//@{
	void initializeGL();
	void resizeGL(int width, int height);
	void paintGL();
	//@}

	bool BindLUT(unsigned int LUTID);

	GLuint LoadShader(const char *FileName, bool VertexFlag);

	/**	@name	Intercept mouse events */
	//@{
	void mouseMoveEvent(QMouseEvent *e);
	void mousePressEvent(QMouseEvent *e);
	void wheelEvent(QWheelEvent*);
	//@}

	/**	@name	Keyboard events */
	//@{
	void keyPressEvent(QKeyEvent *e);
	//@}

	void SetFullScreenMode(bool b);

	protected slots:
		/**	@brief	Update the internal VBOs 
		*
		*	@details
		*	Calling this method will update the internal VBOs to match the current frame. 
		*	Note that this method is supposed to be called within the widgets main thread
		*	in order to comply with the OpenGL render context!
		*	@sa SetTOFData
		*/
		void UpdateVBO(bool SizeChanged);

		/**	@name Mouse rotation */
		//@{
		void SetXRotation(int angle);
		void SetYRotation(int angle);
		void SetZRotation(int angle);
		void SetTranslation(int dx, int dy);
		//@}

		/// Update the current zoom
		void UpdateZoom(float delta);

		/// Reset the camera
		void ResetCamera();

protected:
	/// Flag that indicates whether this widget (in particular the OpenGL extensions are initialized)
	bool m_InitFlag;

	/// Fullscreen flag
	bool m_FullscreenFlag;

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

	/// Flag to control whether the VBOs are ready for use
	bool m_VBOInitialized;
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
};

#endif // CUDAOPENGLVISUALIZATIONWIDGET_H__