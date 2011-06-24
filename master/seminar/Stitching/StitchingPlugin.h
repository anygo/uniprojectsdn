#ifndef StitchingPLUGIN_H__
#define	StitchingPLUGIN_H__

#include "ApplicationPlugin.h"
#include "StitchingPlugin.h"
#include "StitchingWidget.h"

#include "RImage.h"
#include "RImageActorPipeline.h"

typedef ritk::RImageF2					RImageType;
typedef RImageType::Pointer				RImagePointer;
typedef RImageType::ConstPointer		RImageConstPointer;


/**	@class		HistoryListItem
 *	@brief		Extended QListWidgetItem for the history
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that extends QListWidgetItem and holds additional data, in particular
 *	the actual point data inside a RImageActorPipeline and a pointer to the
 *	transformation matrix that was applied to modify the data (during stitching)
 */
class HistoryListItem : public QListWidgetItem
{
public:
	vtkSmartPointer<ritk::RImageActorPipeline>	m_actor;
	vtkSmartPointer<vtkMatrix4x4>				m_transform;
};

/**	@class		StitchingPlugin
 *	@brief		RITK Plugin for (real time) stitching of 3D point clouds
 *	@author		Felix Lugauer and Dominik Neumann
 *
 *	@details
 *	Class that extends ritk::ApplicationPlugin. It controls all user interaction
 *	and encapsulates the whole algorithmic part.
 */
class StitchingPlugin : public ritk::ApplicationPlugin
{
	Q_OBJECT

public:
	/// Constructor
	StitchingPlugin();
	/// Destructor
	~StitchingPlugin();

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


protected:
	StitchingWidget *m_Widget;

	// our functions
	void ExtractValidPoints();
	void Clip(vtkPolyData *toBeClipped);
	void Clean(vtkPolyData *toBeCleaned);
	void LoadFrame();
	void CleanFrame();
	void Stitch(vtkPolyData* toBeStitched, vtkPolyData* previousFrame,
						vtkMatrix4x4* previousTransformationMatrix,
						vtkPolyData* outputStitchedPolyData,
						vtkMatrix4x4* outputTransformationMatrix);

	int m_FramesProcessed;
	QMutex m_Mutex;

	// our members
	vtkSmartPointer<ritk::RImageActorPipeline>	m_DataActor3D;
	ritk::NewFrameEvent::RImageConstPointer		m_CurrentFrame;
	vtkSmartPointer<vtkPolyData>				m_Data;

};


#endif // StitchingPLUGIN_H__
