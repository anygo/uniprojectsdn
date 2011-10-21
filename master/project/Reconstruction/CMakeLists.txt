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


# The headers, sources and Qt related stuff in this plugin
# ========================================================
SET(Reconstruction_HDRS
	ReconstructionPluginFactory.h
	ReconstructionPlugin.h
	ReconstructionWidget.h
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
	optimized ${RITK_COMMON_LIBRARY}
	optimized ${RITK_CORE_LIBRARY}
	${ITK_LIBRARIES} 
	${QT_LIBRARIES}
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
