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

# The headers, sources and Qt related stuff in this plugin
# ========================================================

SET(Reconstruction_HDRS
	ReconstructionPluginFactory.h
	ReconstructionPlugin.h
	ReconstructionWidget.h
	ICP.h
	RBCOneShot.h
)

SET(Reconstruction_SRCS
	ReconstructionPluginFactory.cxx
	ReconstructionPlugin.cxx
	ReconstructionWidget.cxx
)

SET(Reconstruction_MOC_HDRS 	
	ReconstructionPluginFactory.h
	ReconstructionPlugin.h
	ReconstructionWidget.h
)

SET(Reconstruction_UIS
	ReconstructionWidget.ui
)

QT4_WRAP_UI(UI_HDRS ${Reconstruction_UIS})	
QT4_WRAP_CPP(MOC_SRCS ${Reconstruction_MOC_HDRS})


# Set the library name for this plugin
# ====================================
SET(Reconstruction_LIBRARY ReconstructionPlugin CACHE INTERNAL "")
SET(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# Project settings
# ================
SET(Reconstruction_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "")
INCLUDE_DIRECTORIES(${Reconstruction_INCLUDE_DIRS} ${RITK_INCLUDE_DIRS})
ADD_DEFINITIONS(-DQT_PLUGIN)
ADD_DEFINITIONS(-DQT_SHARED)
ADD_LIBRARY(${Reconstruction_LIBRARY} SHARED ${Reconstruction_SRCS} ${Reconstruction_HDRS} ${MOC_SRCS} ${UI_HDRS})
TARGET_LINK_LIBRARIES(${Reconstruction_LIBRARY}
	debug ${RITK_COMMON_LIBRARY_DEBUG}
	debug ${RITK_CORE_LIBRARY_DEBUG}
	debug ${RITK_VTK_LIBRARY_DEBUG}
	debug ${RITK_CUDA_LIBRARY_DEBUG}
	optimized ${RITK_COMMON_LIBRARY}
	optimized ${RITK_CORE_LIBRARY}
	optimized ${RITK_VTK_LIBRARY}
	optimized ${RITK_CUDA_LIBRARY}
	${ITK_LIBRARIES}
	${VTK_LIBRARIES} 
	${QT_LIBRARIES}
	${CUDA_CUDA_LIBRARY} 
	${CUDA_CUDART_LIBRARY} 
	${CUDA_CUT_LIBRARIES}
)
SET_TARGET_PROPERTIES(${Reconstruction_LIBRARY} PROPERTIES DEBUG_POSTFIX "d")


# Module installation
# ===================
INSTALL(TARGETS ${Reconstruction_LIBRARY} CONFIGURATIONS RELEASE
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/release/plugins
	LIBRARY DESTINATION ${RITK_BINARY_DIR}/release/plugins
)
INSTALL(TARGETS ${Reconstruction_LIBRARY} CONFIGURATIONS DEBUG
	RUNTIME DESTINATION ${RITK_BINARY_DIR}/debug/plugins 
	LIBRARY DESTINATION ${RITK_BINARY_DIR}/debug/plugins 
)
