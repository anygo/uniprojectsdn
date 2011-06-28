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
	MESSAGE(FATAL_ERROR "VTK not found, but needed.")
ENDIF(VTK_FOUND)
SET(
	VTK_LIBRARIES
	vtkCommon
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

SET(HDRS
	FastStitchingPluginFactory.h
	FastStitchingPlugin.h
	FastStitchingWidget.h
	CUDAStitcher.h
)

SET(SRCS
	FastStitchingPluginFactory.cxx
	FastStitchingPlugin.cxx
	FastStitchingWidget.cxx
	CUDAStitcher.cxx
)

SET(	CUDA_HDRS
		FastStitchingKernel.h
)

SET(	CUDA_SRCS
		FastStitching.cu
)

SET(  MOC_HDRS 	
	FastStitchingPluginFactory.h
	FastStitchingPlugin.h
	FastStitchingWidget.h
	CUDAStitcher.h
)

SET(  UIS
	FastStitchingWidget.ui
)

QT4_WRAP_UI(UI_HDRS ${UIS})	
QT4_WRAP_CPP(MOC_SRCS ${MOC_HDRS})

#-----------------------------------------------------------------------------

# Set the name for the library
SET(  LIBRARY_NAME
    FastStitchingPlugin)

# Set include directories
SET(  INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
)

# QT plugin
ADD_DEFINITIONS(-DQT_PLUGIN)
ADD_DEFINITIONS(-DQT_SHARED)
#ADD_DEFINITIONS(-DQT_NO_DEBUG)

# Include directories for the plugin
INCLUDE_DIRECTORIES(${INCLUDE_DIRS} ${RITK_INCLUDE_DIRS} ${QT_QTOPENGL_INCLUDE_DIR})
CUDA_ADD_LIBRARY(
  ${LIBRARY_NAME} SHARED
  ${SRCS}
  ${HDRS}
  ${MOC_SRCS}
  ${UI_HDRS}
  ${CUDA_HDRS}
  ${CUDA_SRCS}
)
TARGET_LINK_LIBRARIES(
  ${LIBRARY_NAME} 
	debug ${RITK_Common_LIBRARY_DEBUG}
	debug ${RITK_Core_LIBRARY_DEBUG}
	debug ${RITK_Cuda_LIBRARY_DEBUG}
	debug ${RITK_Visualization_LIBRARY_DEBUG}
	debug ${RITK_Tools_LIBRARY_DEBUG}
	optimized ${RITK_Common_LIBRARY}
	optimized ${RITK_Core_LIBRARY}
	optimized ${RITK_Cuda_LIBRARY}
	optimized ${RITK_Visualization_LIBRARY}
	optimized ${RITK_Tools_LIBRARY}
	${ITK_LIBRARIES}
	${QT_LIBRARIES}
	${VTK_LIBRARIES}
	${QT_QTOPENGL_LIBRARY}
	${CUDA_CUDA_LIBRARY} 
	${CUDA_CUDART_LIBRARY} 
	${CUDA_CUT_LIBRARIES}
)

SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES 
	DEBUG_POSTFIX "d")

INSTALL(TARGETS ${LIBRARY_NAME}
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/release/plugins CONFIGURATIONS RELEASE
)

INSTALL(TARGETS ${LIBRARY_NAME}
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/debug/plugins CONFIGURATIONS DEBUG
)


