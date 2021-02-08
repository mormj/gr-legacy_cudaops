# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT legacy_cudaops_sources)
    MESSAGE(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT legacy_cudaops_sources)

########################################################################
# Check for pygccxml
########################################################################
GR_PYTHON_CHECK_MODULE_RAW(
    "pygccxml"
    "import pygccxml"
    PYGCCXML_FOUND
    )

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND legacy_cudaops_python_files
    memmodel_python.cc 
    copy_python.cc
    cufft_python.cc python_bindings.cc)

GR_PYBIND_MAKE_OOT(legacy_cudaops 
   ../..
   gr::legacy_cudaops
   "${legacy_cudaops_python_files}")

install(TARGETS legacy_cudaops_python DESTINATION ${GR_PYTHON_DIR}/legacy_cudaops COMPONENT pythonapi)
