if(NOT PKG_CONFIG_FOUND)
    INCLUDE(FindPkgConfig)
endif()
PKG_CHECK_MODULES(PC_LEGACY_CUDAOPS legacy_cudaops)

FIND_PATH(
    LEGACY_CUDAOPS_INCLUDE_DIRS
    NAMES legacy_cudaops/api.h
    HINTS $ENV{LEGACY_CUDAOPS_DIR}/include
        ${PC_LEGACY_CUDAOPS_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    LEGACY_CUDAOPS_LIBRARIES
    NAMES gnuradio-legacy_cudaops
    HINTS $ENV{LEGACY_CUDAOPS_DIR}/lib
        ${PC_LEGACY_CUDAOPS_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/legacy_cudaopsTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LEGACY_CUDAOPS DEFAULT_MSG LEGACY_CUDAOPS_LIBRARIES LEGACY_CUDAOPS_INCLUDE_DIRS)
MARK_AS_ADVANCED(LEGACY_CUDAOPS_LIBRARIES LEGACY_CUDAOPS_INCLUDE_DIRS)
