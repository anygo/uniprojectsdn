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
ELSE()
	MESSAGE(FATAL_ERROR "CUDA not found, but needed.")
ENDIF(CUDA_FOUND)

# The headers, sources and Qt related stuff in this plugin
# ========================================================

SET(FastICP_HDRS
	FastICPPluginFactory.h
	FastICPPlugin.h
	FastICPWidget.h
	ICP.h
	RBC.h
	RepGPU.h
	defs.h
	DataGenerator.h
	KinectDataManager.h
	VolumeManager.h
)

SET(FastICP_SRCS
	FastICPPluginFactory.cxx
	FastICPPlugin.cxx
	FastICPWidget.cxx
	ICP.txx
	RBC.txx
	DataGenerator.cxx
	KinectDataManager.cxx
	Instantiations.cxx
	VolumeManager.txx
)

SET(FastICP_CUDA_HDRS
	RBCKernel.h
	ICPKernel.h
	KinectDataManagerKernel.h
	VolumeManagerKernel.h
)

SET(FastICP_CUDA_SRCS
	RBC.cu
	ICP.cu
	KinectDataManager.cu
	VolumeManager.cu
)

SET(FastICP_MOC_HDRS 	
	FastICPPluginFactory.h
	FastICPPlugin.h
	FastICPWidget.h
)

SET(FastICP_UIS
	FastICPWidget.ui
)

QT4_WRAP_UI(UI_HDRS ${FastICP_UIS})	
QT4_WRAP_CPP(MOC_SRCS ${FastICP_MOC_HDRS})


# Set the library name for this plugin
# ====================================
SET(FastICP_LIBRARY FastICPPlugin CACHE INTERNAL "")
SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# Project settings
# ================
SET(FastICP_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
INCLUDE_DIRECTORIES(${FastICP_INCLUDE_DIRS} ${RITK_INCLUDE_DIRS} ${QT_QTOPENGL_INCLUDE_DIR})
ADD_DEFINITIONS(-DQT_PLUGIN)
ADD_DEFINITIONS(-DQT_SHARED)
CUDA_ADD_LIBRARY(${FastICP_LIBRARY} SHARED ${FastICP_SRCS} ${FastICP_HDRS}  ${FastICP_CUDA_HDRS} ${FastICP_CUDA_SRCS} ${MOC_SRCS} ${UI_HDRS})
TARGET_LINK_LIBRARIES(${FastICP_LIBRARY}
	debug ${RITK_COMMON_LIBRARY_DEBUG}
	debug ${RITK_CORE_LIBRARY_DEBUG}
	debug ${RITK_VTK_LIBRARY_DEBUG}
	debug ${RITK_CUDA_LIBRARY_DEBUG}
	debug ${RITK_VISUALIZATION_LIBRARY_DEBUG}
	optimized ${RITK_COMMON_LIBRARY}
	optimized ${RITK_CORE_LIBRARY}
	optimized ${RITK_VTK_LIBRARY}
	optimized ${RITK_CUDA_LIBRARY}
	optimized ${RITK_VISUALIZATION_LIBRARY}
	${QT_QTOPENGL_LIBRARY}
	${ITK_LIBRARIES}
	${VTK_LIBRARIES} 
	${QT_LIBRARIES}
	${CUDA_CUDA_LIBRARY} 
	${CUDA_CUDART_LIBRARY} 
	#${CUDA_CUT_LIBRARIES}
)
SET_TARGET_PROPERTIES(${FastICP_LIBRARY} PROPERTIES DEBUG_POSTFIX "d")


# Module installation
# ===================
INSTALL(TARGETS ${FastICP_LIBRARY} CONFIGURATIONS RELEASE
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/release/plugins
	LIBRARY DESTINATION ${RITK_BINARY_DIR}/release/plugins
)
INSTALL(TARGETS ${FastICP_LIBRARY} CONFIGURATIONS DEBUG
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/debug/plugins 
	LIBRARY DESTINATION ${RITK_BINARY_DIR}/debug/plugins 
)