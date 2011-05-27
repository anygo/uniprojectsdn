cmake_minimum_required(VERSION 2.6)

# ITK
FIND_PACKAGE(ITK REQUIRED)
IF(ITK_FOUND)
	INCLUDE_DIRECTORIES(${ITK_INCLUDE_DIRS})
	INCLUDE(${ITK_USE_FILE})
ELSE()
	MESSAGE("ITK not found, but needed.")
ENDIF(ITK_FOUND)

# VTK
FIND_PACKAGE(VTK REQUIRED)
IF(VTK_FOUND)
	INCLUDE(${VTK_USE_FILE})
	INCLUDE_DIRECTORIES(${VTK_INCLUDE_DIRS})
ELSE()
	MESSAGE("VTK not found, but needed.")
ENDIF(VTK_FOUND)



SET(VTK_LIBRARIES
	vtkCOmmon
	vtkWidgets
	QVTK
)

# Qt4
FIND_PACKAGE(Qt4 REQUIRED)
IF(QT4_FOUND)
	INCLUDE_DIRECTORIES(${QT_INCLUDE_DIR})
	INCLUDE(${QT_USE_FILE})	
ELSE()
	MESSAGE("QT4 not found, but needed.")
ENDIF(QT4_FOUND)

# RITK
FIND_PACKAGE(RITK REQUIRED)
IF(RITK_FOUND)
	INCLUDE_DIRECTORIES(${RITK_INCLUDE_DIR})
	INCLUDE(${RITK_USE_FILE})	
ELSE()
	MESSAGE("RITK not found, but needed.")
ENDIF(RITK_FOUND)




























#-----------------------------------------------------------------------------

SET(Stitching_HDRS
	StitchingPluginFactory.h
	StitchingPlugin.h
	StitchingWidget.h
	defs.h
	ExtendedICPTransform.h
	ClosestPointFinder.h
	ClosestPointFinderBruteForceCPU.h
)

SET(Stitching_SRCS
	StitchingPluginFactory.cxx
	StitchingPlugin.cxx
	StitchingWidget.cxx
	ExtendedICPTransform.cxx
	ClosestPointFinderBruteForceCPU.cxx
)









#-----------------------------------------------------------------------------

# Qt
# all childs of qobject have to be put here
SET(Stitching_MOC_HDRS 	
	StitchingPluginFactory.h
	StitchingPlugin.h
	StitchingWidget.h
)

SET(Stitching_UIS
	StitchingWidget.ui
)

QT4_WRAP_UI(UI_HDRS ${Stitching_UIS})	
QT4_WRAP_CPP(MOC_SRCS ${Stitching_MOC_HDRS})

#-----------------------------------------------------------------------------

# Set the name for the library
SET(Stitching_LIBRARY StitchingPlugin)

# Set include directories
SET(Stitching_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}	
)

# QT plugin
ADD_DEFINITIONS(-DQT_PLUGIN)
ADD_DEFINITIONS(-DQT_SHARED)
#ADD_DEFINITIONS(-DQT_NO_DEBUG)

# Include directories for the plugin
INCLUDE_DIRECTORIES(${Stitching_INCLUDE_DIRS} ${RITK_INCLUDE_DIRS})
ADD_LIBRARY(${Stitching_LIBRARY} SHARED ${Stitching_SRCS} ${Stitching_HDRS} ${MOC_SRCS} ${UI_HDRS})









TARGET_LINK_LIBRARIES(${Stitching_LIBRARY} 

	debug ${RITK_Common_LIBRARY_DEBUG}
	debug ${RITK_Core_LIBRARY_DEBUG}

	debug ${RITK_Visualization_LIBRARY_DEBUG}

	optimized ${RITK_Common_LIBRARY}
	optimized ${RITK_Core_LIBRARY}

	optimized ${RITK_Visualization_LIBRARY}

	${ITK_LIBRARIES} ${QT_LIBRARIES} ${VTK_LIBRARIES}







	)

SET_TARGET_PROPERTIES(${Stitching_LIBRARY} PROPERTIES 
	DEBUG_POSTFIX "d")

INSTALL(TARGETS ${Stitching_LIBRARY}
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/release/plugins CONFIGURATIONS RELEASE
)

INSTALL(TARGETS ${Stitching_LIBRARY}
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/debug/plugins CONFIGURATIONS DEBUG
)


