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
	void LoadFrame(bool update = true);
	void CleanFrame(bool update = true);
	void StitchToWorld(bool update = true);
	void LoadCleanStitch();
	void CleanWorld();
	void Delaunay2D();
	void SaveVTKData();
	void InitializeWorld();


protected:
	StitchingWidget *m_Widget;

	// our functions
	void ExtractPointcloud();
	void Clip(vtkPolyData *toBeClipped);
	void Clean(vtkPolyData *toBeCleaned);

	// our members
	vtkSmartPointer<ritk::RImageActorPipeline>							m_DataActor3D;
	ritk::NewFrameEvent::RImageType::RangeImageType::ConstPointer		m_RangeImage;
	ritk::NewFrameEvent::RImageType::RGBImageType::ConstPointer			m_RGBImage;
	ritk::NewFrameEvent::RImageConstPointer								m_CurrentFrame;
	vtkSmartPointer<ritk::RImageVTKData>								m_RImageVTKData;
	vtkSmartPointer<vtkPolyData>										m_PolyData;
	vtkSmartPointer<vtkPolyData>										m_Data;
	vtkSmartPointer<vtkPolyData>										m_TheWorld;
	vtkSmartPointer<vtkPolyData>										m_PreviousFrame;
	vtkSmartPointer<vtkMatrix4x4>										m_PreviousTransform;

};


#endif // StitchingPLUGIN_H__
