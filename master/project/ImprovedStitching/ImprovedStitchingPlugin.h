#ifndef IMPROVEDSTITCHINGPLUGIN_H__
#define	IMPROVEDSTITCHINGPLUGIN_H__

#include "ritkApplicationPlugin.h"
#include "ritkRImage.h"
#include "ritkOpenGLRGBAVolumeCudaRayCastEntity.h"

#include "ImprovedStitchingWidget.h"

#include "defs.h"
#include "ICP.h"
#include "KinectDataManager.h"
#include "VolumeManager.h"


/**	@class		ImprovedStitchingPlugin
 *	@author		Dominik Neumann
 *	@brief		Stitching of multiple 3D point clouds (using an occupancy grid)
 *
 *	@details 
 *	Plugin that internally uses our efficient ICP and its RBC variant to 'stitch' 3D point
 *	cloud streams. For visualization, an occupancy grid based approach is used.
 */
class ImprovedStitchingPlugin : public ritk::ApplicationPlugin
{
	Q_OBJECT

public:
	/**	@name Plugin Typedefs */
	//@{
	typedef ritk::RImageF2																			RImageType;
	typedef RImageType::Pointer																		RImagePointer;
	typedef RImageType::ConstPointer																RImageConstPointer;

	typedef ritk::OpenGLRGBAVolumeCudaRayCastEntity													RayCastEntityType;
	typedef ritk::Cuda3DArrayImportImageContainer<ulong, RayCastEntityType::VolumeType::PixelType>	Cuda3DArrayContainerType;
	//@}

	/// Constructor
	ImprovedStitchingPlugin();

	/// Destructor
	~ImprovedStitchingPlugin();

	/// Get the name @sa Plugin::GetName
	QString GetName();

	/// Get the plugins GUI @sa Plugin::GetPluginGUI
	QWidget *GetPluginGUI();

	/// Intercept the event.
	void ProcessEvent(ritk::Event::Pointer EventP);

signals:
	void UpdateGUI();

	void FrameToStitchAvailable(bool Visualize);

protected slots:

	/// Call constructor for current ICP object and delete previous one
	void ChangeNumPts();

	/// Apply new weights (geometric vs. photometric information)
	void ChangeRGBWeight(int Value);

	/// Set clip percentage to Value (for moving point set)
	void ChangeClipPercentage(int Value);

	/// Save internal volume as meta image file (.mha)
	void SaveVolume();

	/// Reset volume to initial state (all voxels -> 0)
	void ResetVolume();

	// Stitch the current frame to the volume
	void AutoStitch(bool Visualize);

protected:
	/// Checks for existing ICP objects and deletes them as appropriate
	void DeleteICPObjects();

	/// Run ICP on a given set of moving and fixed points
	void RunICP();

	/// Stores weight for payload (RGB data), used during NN search (RBC)
	float m_PayloadWeight;

	/// Stores currently selected value in combo box for NumPts
	unsigned long m_NumPts;

	/**	@name Pointers to ICP objects using N points */
	//@{
	ICP<512,	ICP_DATA_DIM>* m_ICP512;
	ICP<1024,	ICP_DATA_DIM>* m_ICP1024;
	ICP<2048,	ICP_DATA_DIM>* m_ICP2048;
	ICP<4096,	ICP_DATA_DIM>* m_ICP4096;
	ICP<8192,	ICP_DATA_DIM>* m_ICP8192;
	ICP<16384,	ICP_DATA_DIM>* m_ICP16384;
	//@}

	/// Pointer to current frame from RITK pipeline
	ritk::NewFrameEvent::RImageConstPointer m_CurrentFrame;

	/**	@name Pointers to Kinect data managers (required when we use real data) */
	//@{
	itk::SmartPointer<KinectDataManager> m_KinectFixed;
	itk::SmartPointer<KinectDataManager> m_KinectMoving;
	//@}

	/// The volume manager
	VolumeManager::Pointer m_Volume;

	/// The volume entity
	RayCastEntityType::Pointer m_VolumeRayCastEntity;

	/// Volume for raycasting
	RayCastEntityType::VolumeType::Pointer m_VolumeForRaycasting;

	/// Indicator for auto stitching behavior
	bool m_IsFirstFrame;

	/// The widget
	ImprovedStitchingWidget *m_Widget;
};


#endif // IMPROVEDSTITCHINGPLUGIN_H__
