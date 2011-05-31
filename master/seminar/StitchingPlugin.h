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


// defs
class HistoryListItem : public QListWidgetItem
{
public:
	vtkSmartPointer<ritk::RImageActorPipeline>	m_actor;
	vtkSmartPointer<vtkMatrix4x4>				m_transform;
};


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

protected slots:
	void LoadFrame();
	void CleanFrame();
	void Stitch();
	void LoadCleanStitch();
	void LoadCleanInitialize();
	void Delaunay2D();
	void SaveVTKData();
	void InitializeHistory();
	void ChangePointSize();
	void ChangeBackgroundColor1();
	void ChangeBackgroundColor2();
	void ShowHideActors();
	void DeleteSelectedActors();
	void MergeSelectedActors();
	void CleanSelectedActors();
	void HighlightActor(QListWidgetItem*);


protected:
	StitchingWidget *m_Widget;

	// our functions
	void ExtractValidPoints();
	void Clip(vtkPolyData *toBeClipped);
	void Clean(vtkPolyData *toBeCleaned);

	int m_FramesProcessed;

	// our members
	vtkSmartPointer<ritk::RImageActorPipeline>	m_DataActor3D;
	ritk::NewFrameEvent::RImageConstPointer		m_CurrentFrame;
	vtkSmartPointer<vtkPolyData>				m_Data;

};


#endif // StitchingPLUGIN_H__
