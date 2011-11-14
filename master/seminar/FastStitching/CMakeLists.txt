cmake_minimum_required(VERSION 2.6)

# ITK
FIND_PACKAGE(ITK REQUIRED)
IF(ITK_FOUND)
	INCLUDE_DIRECTORIES(${ITK_INCLUDE_DIRS})
	INCLUDE(${ITK_USE_FILE})
ELSE()
	MESSAGE("ITK not found, but needed.")
ENDIF(ITK_FOUND)

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


# VTK
FIND_PACKAGE(VTK REQUIRED)
IF(VTK_FOUND)
	INCLUDE(${VTK_USE_FILE})
	INCLUDE_DIRECTORIES(${VTK_INCLUDE_DIRS})
ELSE()
	MESSAGE(FATAL_ERROR "VTK not found, but needed.")
ENDIF(VTK_FOUND)

SET(VTK_LIBRARIES
	vtkCOmmon
	vtkWidgets
	QVTK
)



# CUDA
FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
	FIND_PATH(CUDA_CUT_INCLUDE_DIR
		cutil.h
		PATHS ${CUDA_SDK_SEARCH_PATH}
		PATH_SUFFIXES "common/inc"
		DOC "Location of cutil.h"
		NO_DEFAULT_PATH
    	)
	mark_as_advanced(CUDA_CUT_INCLUDE_DIR)
	INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
	
	FIND_LIBRARY(CUDA_CUT_LIBRARY
		NAMES cutil ${cuda_cutil_name}
		PATHS ${CUDA_SDK_SEARCH_PATH}
		PATH_SUFFIXES "common/lib" "lib"
		DOC "Location of cutil library"
		NO_DEFAULT_PATH
	)
	#find_library(CUDA_CUT_LIBRARY NAMES cutil ${cuda_cutil_name} DOC "Location of cutil library")
	mark_as_advanced(CUDA_CUT_LIBRARY)
	set(CUDA_CUT_LIBRARIES ${CUDA_CUT_LIBRARY})
ELSE()
	MESSAGE(FATAL_ERROR "CUDA not found, but needed.")
ENDIF(CUDA_FOUND)


#-----------------------------------------------------------------------------
# Set headers (HDRS), sources (SRCS), childs of QObject (MOC_HDRS), and Qt ui-files (UIS)

SET(FastStitching_HDRS
	FastStitchingPluginFactory.h
	FastStitchingPlugin.h
	FastStitchingWidget.h
	defs.h
	ExtendedICPTransform.h
	ClosestPointFinder.h
	ClosestPointFinderRBCGPU.h
)

SET(FastStitching_SRCS
	FastStitchingPluginFactory.cxx
	FastStitchingPlugin.cxx
	FastStitchingWidget.cxx
	ExtendedICPTransform.cxx
	ClosestPointFinderRBCGPU.cxx
)

SET(FastStitching_CUDA_HDRS
	FastStitchingPluginKernel.h
)

SET(FastStitching_CUDA_SRCS
	FastStitchingPlugin.cu
)

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


# Set the library name for this plugin
# ====================================
SET(FastStitching_LIBRARY FastStitchingPlugin CACHE INTERNAL "")
SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#CUDA_ADD_LIBRARY(${FastStitching_LIBRARY} SHARED
#	${FastStitching_CUDA_HDRS} 
#	${FastStitching_CUDA_SRCS}
#)

# Project settings
# ================
SET(FastStitching_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
INCLUDE_DIRECTORIES(${FastStitching_INCLUDE_DIRS} ${RITK_INCLUDE_DIRS})
ADD_DEFINITIONS(-DQT_PLUGIN)
ADD_DEFINITIONS(-DQT_SHARED)
ADD_LIBRARY(${FastStitching_LIBRARY} SHARED ${FastStitching_SRCS} ${FastStitching_HDRS} ${FastStitching_CUDA_HDRS} ${FastStitching_CUDA_SRCS} ${MOC_SRCS} ${UI_HDRS})
TARGET_LINK_LIBRARIES(${FastStitching_LIBRARY}
	debug ${RITK_COMMON_LIBRARY_DEBUG}
	debug ${RITK_CORE_LIBRARY_DEBUG}
	debug ${RITK_VISUALIZATION_LIBRARY_DEBUG}
	debug ${RITK_CUDA_LIBRARY_DEBUG}
	debug ${RITK_TOOLS_LIBRARY_DEBUG}
	optimized ${RITK_COMMON_LIBRARY}
	optimized ${RITK_CORE_LIBRARY}
	optimized ${RITK_VISUALIZATION_LIBRARY}
	optimized ${RITK_CUDA_LIBRARY}
	optimized ${RITK_TOOLS_LIBRARY}
	${ITK_LIBRARIES} 
	${QT_LIBRARIES}
	${VTK_LIBRARIES}
	${CUDA_CUDA_LIBRARY} 
	${CUDA_CUDART_LIBRARY} 
	${CUDA_CUT_LIBRARIES}
)
SET_TARGET_PROPERTIES(${FastStitching_LIBRARY} PROPERTIES DEBUG_POSTFIX "d")


#TARGET_LINK_LIBRARIES(
#	${LIBRARY_NAME} 
#	debug ${RITK_Common_LIBRARY_DEBUG}
#	debug ${RITK_Core_LIBRARY_DEBUG}
#	debug ${RITK_Visualization_LIBRARY_DEBUG}
#	debug ${RITK_Cuda_LIBRARY_DEBUG}
#	optimized ${RITK_Common_LIBRARY}
#	optimized ${RITK_Core_LIBRARY}
#	optimized ${RITK_Visualization_LIBRARY}
#	optimized ${RITK_Cuda_LIBRARY}
#	optimized ${RITK_Tools_LIBRARY}
#	${ITK_LIBRARIES} 
#	${QT_LIBRARIES}	
#	${QT_QTOPENGL_LIBRARY}	
#)


# Module installation
# ===================
INSTALL(TARGETS ${FastStitching_LIBRARY} CONFIGURATIONS RELEASE
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/release/plugins
	LIBRARY DESTINATION ${RITK_BINARY_DIR}/release/plugins
)
INSTALL(TARGETS ${FastStitching_LIBRARY} CONFIGURATIONS DEBUG
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/debug/plugins 
	LIBRARY DESTINATION ${RITK_BINARY_DIR}/debug/plugins 
)
