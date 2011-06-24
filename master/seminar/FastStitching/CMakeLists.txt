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

SET(FastStitching_HDRS
	FastStitchingPluginFactory.h
	FastStitchingPlugin.h
	FastStitchingWidget.h
)

SET(FastStitching_SRCS
	FastStitchingPluginFactory.cxx
	FastStitchingPlugin.cxx
	FastStitchingWidget.cxx
)

#-----------------------------------------------------------------------------

# Qt
# all childs of qobject have to be put here
SET(FastStitching_MOC_HDRS 	
	FastStitchingPluginFactory.h
	FastStitchingPlugin.h
	FastStitchingWidget.h
)

SET(FastStitching_UIS
	FastStitchingWidget.ui
)

QT4_WRAP_UI(UI_HDRS ${FastStitching_UIS})	
QT4_WRAP_CPP(MOC_SRCS ${FastStitching_MOC_HDRS})

#-----------------------------------------------------------------------------

# Set the name for the library
SET(FastStitching_LIBRARY FastStitchingPlugin)

# Set include directories
SET(FastStitching_INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}	
)

# QT plugin
ADD_DEFINITIONS(-DQT_PLUGIN)
ADD_DEFINITIONS(-DQT_SHARED)
#ADD_DEFINITIONS(-DQT_NO_DEBUG)

# Include directories for the plugin
INCLUDE_DIRECTORIES(${FastStitching_INCLUDE_DIRS} ${RITK_INCLUDE_DIRS})
ADD_LIBRARY(${FastStitching_LIBRARY} SHARED ${FastStitching_SRCS} ${FastStitching_HDRS} ${MOC_SRCS} ${UI_HDRS})
TARGET_LINK_LIBRARIES(${FastStitching_LIBRARY} 
	debug ${RITK_Common_LIBRARY_DEBUG}
	debug ${RITK_Core_LIBRARY_DEBUG}
	debug ${RITK_Visualization_LIBRARY_DEBUG}
	optimized ${RITK_Common_LIBRARY}
	optimized ${RITK_Core_LIBRARY}
	optimized ${RITK_Visualization_LIBRARY}
	${ITK_LIBRARIES} ${QT_LIBRARIES} ${VTK_LIBRARIES}
	)

SET_TARGET_PROPERTIES(${FastStitching_LIBRARY} PROPERTIES 
	DEBUG_POSTFIX "d")

INSTALL(TARGETS ${FastStitching_LIBRARY}
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/release/plugins CONFIGURATIONS RELEASE
)

INSTALL(TARGETS ${FastStitching_LIBRARY}
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/debug/plugins CONFIGURATIONS DEBUG
)


