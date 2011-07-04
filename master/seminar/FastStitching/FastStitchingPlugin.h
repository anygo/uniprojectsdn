#ifndef FastStitchingPLUGIN_H__
#define	FastStitchingPLUGIN_H__

#include "ApplicationPlugin.h"
#include "FastStitchingPlugin.h"
#include "FastStitchingWidget.h"

#include "RImage.h"
#include "RImageActorPipeline.h"
#include <ClosestPointFinder.h>
#include <ExtendedICPTransform.h>
#include <cutil_inline.h>

typedef ritk::RImageF2					RImageType;
typedef RImageType::Pointer				RImagePointer;
typedef RImageType::ConstPointer		RImageConstPointer;


/**	@class		HistoryListItem
 *	@brief		Extended QListWidgetItem for the history
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that extends QListWidgetItem and holds additional data, in particular
 *	the actual point data inside a RImageActorPipeline and the transformation 
 *	matrix that was applied to modify the data (during stitching)
 */
class HistoryListItem : public QListWidgetItem
{
public:
	vtkSmartPointer<ritk::RImageActorPipeline>	m_actor;
	vtkSmartPointer<vtkMatrix4x4>				m_transform;
};

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
	void UpdateProgressBar(int val);
	void InitProgressBar(int from, int to);
	void UpdateStats();
	void RecordFrameAvailable();
	void LiveFastStitchingFrameAvailable();

protected slots:
	void LoadStitch();
	void LoadInitialize();
	void Delaunay2DSelectedActors();
	void SaveSelectedActors();
	void InitializeHistory();
	void ChangePointSize();
	void ChangeBackgroundColor1();
	void ChangeBackgroundColor2();
	void ShowHideActors();
	void DeleteSelectedActors();
	void MergeSelectedActors();
	void CleanSelectedActors();
	void StitchSelectedActors();
	void UndoTransformForSelectedActors();
	void HighlightActor(QListWidgetItem*);
	void ComputeStats();
	void ResetICPandCPF(); 
	void RecordFrame();
	void LiveFastStitching();
	void ClearBuffer();
	void UpdateZRange();
	void SetThreshold(double thresh) { m_HistogramDifferenceThreshold = thresh; }
	void ComputeStatistics();


protected:
	FastStitchingWidget *m_Widget;

	// our functions
	void Clean(vtkPolyData *toBeCleaned);
	void LoadFrame();
	bool FrameDifferenceAboveThreshold();
	void CleanFrame();
	void Stitch(vtkPolyData* toBeStitched, vtkPolyData* previousFrame,
						vtkMatrix4x4* previousTransformationMatrix,
						vtkPolyData* outputStitchedPolyData,
						vtkMatrix4x4* outputTransformationMatrix);

	
	// our members
	ritk::NewFrameEvent::RImageConstPointer		m_CurrentFrame;
	float*										m_CurrentHist;
	float*										m_PreviousHist;
	vtkSmartPointer<vtkPolyData>				m_Data;
	vtkSmartPointer<ExtendedICPTransform>		m_icp;
	ClosestPointFinder*							m_cpf; 
	float4*										m_devWCs;
	float4*										m_WCs;
	unsigned char*								m_RangeTextureData;
	cudaArray*									m_InputImgArr;

	int m_FramesProcessed;
	QMutex m_Mutex;
	int m_BufferSize;
	int m_BufferCounter;
	bool m_ResetICPandCPFRequired;
	float m_MinZ;
	float m_MaxZ;
	float m_HistogramDifferenceThreshold;
};


#endif // FastStitchingPLUGIN_H__
