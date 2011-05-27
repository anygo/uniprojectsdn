cmake_minimum_required(VERSION 2.6)

#-----------------------------------------------------------------------------
# Required libraries

# ITK
FIND_PACKAGE(ITK REQUIRED)
IF(ITK_FOUND)
	INCLUDE_DIRECTORIES(${ITK_INCLUDE_DIRS})
	INCLUDE(${ITK_USE_FILE})
ELSE()
	MESSAGE(FATAL_ERROR "ITK not found, but needed.")
ENDIF(ITK_FOUND)

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

# boost
FIND_PACKAGE(BOOST REQUIRED)
#IF(BOOST_FOUND)
	LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
	INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
#ELSE()
	#MESSAGE(FATAL_ERROR "Boost not found, but needed.")
#ENDIF(BOOST_FOUND)

# Qt4
FIND_PACKAGE(Qt4 REQUIRED)
IF(QT4_FOUND)
	INCLUDE_DIRECTORIES(${QT_INCLUDE_DIR})
	INCLUDE(${QT_USE_FILE})	
ELSE()
	MESSAGE(FATAL_ERROR "QT4 not found, but needed.")
ENDIF(QT4_FOUND)

# RITK
FIND_PACKAGE(RITK REQUIRED)
IF(RITK_FOUND)
	INCLUDE_DIRECTORIES(${RITK_INCLUDE_DIR})
	INCLUDE(${RITK_USE_FILE})	
ELSE()
	MESSAGE(FATAL_ERROR "RITK not found, but needed.")
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
# Set headers (HDRS), sources (SRCS), childs of QObject (MOC_HDRS), and Qt ui-files (UIS)

SET(HDRS
	StitchingPluginFactory.h
	StitchingPlugin.h
	StitchingWidget.h
	defs.h
	ExtendedICPTransform.h
	ClosestPointFinder.h
	ClosestPointFinderBruteForceCPU.h
)

SET(SRCS
	StitchingPluginFactory.cxx
	StitchingPlugin.cxx
	StitchingWidget.cxx
	ExtendedICPTransform.cxx
	ClosestPointFinderBruteForceCPU.cxx
)


SET(CUDA_HDRS
	CudaTestKernel.h
)


SET(CUDA_SRCS
	CudaTest.cu
)

#-----------------------------------------------------------------------------

# Qt
# all childs of qobject have to be put here
SET(MOC_HDRS 	
	StitchingPluginFactory.h
	StitchingPlugin.h
	StitchingWidget.h
)

SET(UIS
	StitchingWidget.ui
)

QT4_WRAP_UI(UI_HDRS ${UIS})	
QT4_WRAP_CPP(MOC_SRCS ${MOC_HDRS})

#-----------------------------------------------------------------------------

# Set the name for the library
SET(LIBRARY_NAME 
	StitchingPlugin
)

SET(INCLUDE_DIRS
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}	
)


#-----------------------------------------------------------------------------
# Set include directories, definitions and libs for this plugin

INCLUDE_DIRECTORIES(
	${INCLUDE_DIRS} 
	${RITK_INCLUDE_DIRS}
	${QT_QTOPENGL_INCLUDE_DIR}
)
ADD_LIBRARY(
	${LIBRARY_NAME} SHARED 
	${SRCS} 
	${HDRS} 
	${CUDA_HDRS} 
	${CUDA_SRCS}
	${MOC_SRCS} 
	${UI_HDRS}
)
TARGET_LINK_LIBRARIES(
	${LIBRARY_NAME} 
	debug ${RITK_Common_LIBRARY_DEBUG}
	debug ${RITK_Core_LIBRARY_DEBUG}
	debug ${RITK_Visualization_LIBRARY_DEBUG}
	debug ${RITK_Cuda_LIBRARY_DEBUG}
	optimized ${RITK_Common_LIBRARY}
	optimized ${RITK_Core_LIBRARY}
	optimized ${RITK_Visualization_LIBRARY}
	optimized ${RITK_Cuda_LIBRARY}
	optimized ${RITK_Tools_LIBRARY}
	${ITK_LIBRARIES} 
	${QT_LIBRARIES}
	${VTK_LIBRARIES}
	${QT_QTOPENGL_LIBRARY}
	${CUDA_CUDA_LIBRARY} 
	${CUDA_CUDART_LIBRARY} 
	${CUDA_CUT_LIBRARIES}
)
SET_TARGET_PROPERTIES(
	${LIBRARY_NAME} 
	PROPERTIES  DEBUG_POSTFIX "d"
)
ADD_DEFINITIONS(-DQT_PLUGIN)
ADD_DEFINITIONS(-DQT_SHARED)


#-----------------------------------------------------------------------------
# Installation
INSTALL(TARGETS ${LIBRARY_NAME} RUNTIME DESTINATION ${RITK_BINARY_DIR}/release/plugins CONFIGURATIONS RELEASE)
INSTALL(TARGETS ${LIBRARY_NAME} RUNTIME DESTINATION ${RITK_BINARY_DIR}/debug/plugins CONFIGURATIONS DEBUG)
