/*
 * Copyright 2021 Free Software Foundation, Inc.
 *
 * This file is part of GNU Radio
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 */

/***********************************************************************************/
/* This file is automatically generated using bindtool and can be manually edited  */
/* The following lines can be configured to regenerate this file during cmake      */
/* If manual edits are made, the following tags should be modified accordingly.    */
/* BINDTOOL_GEN_AUTOMATIC(0)                                                       */
/* BINDTOOL_USE_PYGCCXML(0)                                                        */
/* BINDTOOL_HEADER_FILE(memmodel.h)                                        */
/* BINDTOOL_HEADER_FILE_HASH(0d5eb907f7f402869c80c154f1fa0970)                     */
/***********************************************************************************/

#include <pybind11/complex.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <legacy_cudaops/memmodel.h>
// pydoc.h is automatically generated in the build directory
#include <memmodel_pydoc.h>

void bind_memmodel(py::module& m)
{


    py::enum_<::gr::legacy_cudaops::memory_model_t>(m,"memory_model_t")
        .value("TRADITIONAL", ::gr::legacy_cudaops::memory_model_t::TRADITIONAL) // 0
        .value("PINNED", ::gr::legacy_cudaops::memory_model_t::PINNED) // 1
        .value("UNIFIED", ::gr::legacy_cudaops::memory_model_t::UNIFIED) // 2
        .export_values()
    ;

    py::implicitly_convertible<int, ::gr::legacy_cudaops::memory_model_t>();



}








