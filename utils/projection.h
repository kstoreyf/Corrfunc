/* File: projection.h */
/*
 *   This file is a part of the Corrfunc package
 *     Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
 *       License: MIT LICENSE. See LICENSE file under the top-level
 *         directory at https://github.com/manodeep/Corrfunc/
 *         */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include "defs.h"

void compute_amplitudes(int nprojbins, int nd1, int nd2, int nr1, int nr2,
            void *dd, void *dr, void *rd, void *rr, void *qq, void *amps, size_t element_size);

void evaluate_xi(int nprojbins, void *amps, int nsvals, void *svals, int nsbins, 
          void *sbins, void *xi, proj_method_t proj_method, size_t element_size, char *projfn);
