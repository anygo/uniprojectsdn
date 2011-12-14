#ifndef FASTICPPLUGIN_H__
#define	FASTICPPLUGIN_H__

#include "ritkApplicationPlugin.h"
#include "ritkRImage.h"
#include "ritkRImageActorPipeline.h"

#include "FastICPPlugin.h"
#include "FastICPWidget.h"

#include "defs.h"
#include "ICP.h"
#include "DataGenerator.h"
#include "KinectDataManager.h"


typedef ritk::RImageF2								RImageType;
typedef RImageType::Pointer							RImagePointer;
typedef RImageType::ConstPointer					RImageConstPointer;
typedef vtkSmartPointer<ritk::RImageActorPipeline>	ActorPointer;


/**	@class	FastICPPlugin
 *	@author	Dominik Neumann
 *	@brief	Sample application using our efficient ICP variant
 *
 *	@details
 *	This application plugin allows to generate synthetic data to test our
 *	Random Ball Cover based ICP implementation
 */
class FastICPPlugin : public ritk::ApplicationPlugin
{
	Q_OBJECT

public:
	/// Constructor
	FastICPPlugin();
	/// Destructor
	~FastICPPlugin();

	/// Get the name @sa Plugin::GetName
	QString GetName();

	/// Get the plugins GUI @sa Plugin::GetPluginGUI
	QWidget *GetPluginGUI();

	/// Intercept the event.
	void ProcessEvent(ritk::Event::Pointer EventP);

signals:
	void UpdateGUI();

protected slots:
	/// Run ICP on a given set of moving and fixed points
	void RunICP();

	/// Call constructor for current ICP object and delete previous one
	void ChangeNumPts();

	/// Wrapper for PlotCorrespondenceLines that checks which ICP is currently in use fetches the data from that object
	void PlotCorrespondenceLinesWrapper();

	/// Apply new weights (geometric vs. photometric information)
	void ChangeRGBWeight(int Value);

	/// Generate synthetic data (either Gaussian distributed or uniformly distributed points)
	void GenerateSyntheticData();

	/// Import set of fixed points from Kinect image
	void ImportFixedData();

	/// Import set of moving points from Kinect image
	void ImportMovingData();

	/// Set clip percentage to Value (for moving point set)
	void ChangeClipPercentage(int Value);

	/// Called when user clicks on check box for "Show Landmarks"
	void ToggleShowLandmarks();

	/// This method is called when the user toggles the data mode (synthetic -> Kinect data or vice versa)
	void ToggleDataMode();

protected:
	/// The widget
	FastICPWidget *m_Widget;

	/// Checks for existing ICP objects and deletes them as appropriate
	void DeleteICPObjects();

	/// Helper function that converts float* Data into polyData and puts it into Actor
	void CopyFloatDataToActor(float* Data, unsigned long NumPts, ActorPointer Actor);

	/// Generates lines between corresponding points in the given datasets (assumes points with same index in both sets correspond to each other)
	void PlotCorrespondenceLines(float* Data1, float* Data2, ActorPointer Actor);

	/**	@name Pointers to actors for fixed and moving point set, used for visualization */
	//@{
	ActorPointer m_ActorFixed;
	ActorPointer m_ActorMoving;
	ActorPointer m_ActorLines;
	//@}

	/// Stores weight for payload (RGB data), used during NN search (RBC)
	float m_PayloadWeight;

	/// Stores currently selected value in combo box for NumPts
	unsigned long m_NumPts;

	/**	@name Pointers to ICP objects using N points */
	//@{
	ICP<32,		ICP_DATA_DIM>* m_ICP32;
	ICP<512,	ICP_DATA_DIM>* m_ICP512;
	ICP<1024,	ICP_DATA_DIM>* m_ICP1024;
	ICP<2048,	ICP_DATA_DIM>* m_ICP2048;
	ICP<4096,	ICP_DATA_DIM>* m_ICP4096;
	ICP<8192,	ICP_DATA_DIM>* m_ICP8192;
	ICP<16384,	ICP_DATA_DIM>* m_ICP16384;
	//@}
	
	/// Pointer to our data generator for synthetic data
	DataGenerator* m_DataGenerator;

	/// Pointer to current frame from RITK pipeline
	ritk::NewFrameEvent::RImageConstPointer m_CurrentFrame;

	/**	@name Pointers to Kinect data managers (required when we use real data) */
	//@{
	KinectDataManager* m_KinectFixed;
	KinectDataManager* m_KinectMoving;
	//@}

	/// True, if user chooses to show landmarks instead of the whole frame (Kinect data only)
	bool m_ShowLandmarks;

	/// ICP can only work if two sets of points are imported
	bool m_KinectFixedImported;

	/// ICP can only work if two sets of points are imported
	bool m_KinectMovingImported;

	/// Indicates, whether at least one frame passed ProcessEvent()
	bool m_FrameAvailable;

	/// True, if we currently work on synthetic data, otherwise it will be false (Kinect data)
	bool m_SyntheticDataMode;
};


#endif // FASTICPPLUGIN_H__
