#ifndef FastStitchingPLUGIN_H__
#define	FastStitchingPLUGIN_H__

#include "ritkApplicationPlugin.h"
#include "FastStitchingPlugin.h"
#include "FastStitchingWidget.h"

#include "ritkRImage.h"
#include "ritkRImageActorPipeline.h"
#include <ClosestPointFinder.h>
#include <ExtendedICPTransform.h>
#include <cutil_inline.h>

typedef ritk::RImageF2					RImageType;
typedef RImageType::Pointer				RImagePointer;
typedef RImageType::ConstPointer		RImageConstPointer;


/**	@class		FastStitchingPlugin
 *	@brief		RITK Plugin for (real time) stitching of 3D point clouds
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that extends ritk::ApplicationPlugin. It controls all user interaction
 *	and encapsulates the whole algorithmic part.
 */
class FastStitchingPlugin : public ritk::ApplicationPlugin
{
	Q_OBJECT

public:
	/// Constructor
	FastStitchingPlugin();

	/// Destructor
	~FastStitchingPlugin();

	/// Get the name @sa Plugin::GetName
	QString GetName();

	/// Get the plugins GUI @sa Plugin::GetPluginGUI
	QWidget *GetPluginGUI();

	/// Intercept the event.
	void ProcessEvent(ritk::Event::Pointer EventP);

signals:
	void UpdateGUI();
	void UpdateStats();
	void RecordFrameAvailable();
	void NewFrameAvailable();

protected slots:
	void LoadStitch();
	void ResetICPandCPF() { m_ResetICPandCPFRequired = true; } 


protected:
	FastStitchingWidget *m_Widget;

	// our functions
	void LoadFrame();
	bool FrameDifferenceAboveThreshold();
	void Reset();
	void ExtractLandmarks();
	void Stitch();
	void CopyToCPUAndVisualizeFrame();

	
	// our members
	bool m_FirstFrame;
	ritk::NewFrameEvent::RImageConstPointer		m_CurrentFrame;
	ExtendedICPTransform*						m_icp;
	ClosestPointFinder*							m_cpf; 
	float4*										m_devWCs;
	float4*										m_devPrevWCs;
	float4*										m_WCs;
	uchar3*										m_devColors;
	uchar3*										m_devPrevColors;

	unsigned char*								m_RangeTextureData;
	cudaArray*									m_InputImgArr;
	unsigned int*								m_ClippedLMIndices;
	unsigned int*								m_LMIndices;

	unsigned int*								m_SrcIndices;
	unsigned int*								m_TargetIndices;

	unsigned int*								m_devSourceIndices;
	unsigned int*								m_devTargetIndices;

	float4*										m_devSourceLandmarks;
	float4*										m_devTargetLandmarks;
	float4*										m_devCurLandmarksColor;
	float4*										m_devPrevLandmarksColor;

	int											m_FramesProcessed;
	QMutex										m_Mutex;
	bool										m_ResetICPandCPFRequired;
	int											m_NumLandmarks;

	vtkSmartPointer<vtkMatrix4x4>				m_PreviousTransform;
};


#endif // FastStitchingPLUGIN_H__
