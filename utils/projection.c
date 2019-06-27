/* File: projection.c */
/*
 *   This file is a part of the Corrfunc package
 *     Copyright (C) 2015-- Manodeep Sinha (manodeep@gmail.com)
 *       License: MIT LICENSE. See LICENSE file under the top-level
 *         directory at https://github.com/manodeep/Corrfunc/
 *         */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "defs.h"
#include "projection.h" //function proto-type for API
#include "proj_functions_double.h"//actual implementations for double
#include "proj_functions_float.h"//actual implementations for float


int compute_amplitudes(int nprojbins, int nd1, int nd2, int nr1, int nr2,
            void *dd, void *dr, void *rd, void *rr, void *qq, void *amps, size_t element_size)
{
    printf("projection\n");
    if( ! (element_size == sizeof(float) || element_size == sizeof(double))){
        fprintf(stderr,"ERROR: In %s> Can only handle doubles or floats. Got an array of size = %zu\n",
                __FUNCTION__, element_size);
        return EXIT_FAILURE;
    }
    printf("elementsize: %d, float: %d, double: %d\n", element_size, sizeof(float), sizeof(double));
    if(element_size == sizeof(float)) {
        return compute_amplitudes_float(nprojbins, nd1, nd2, nr1, nr2,
            (float *) dd, (float *) dr, (float *) rd, (float *) rr, (float *) qq, (float *) amps);
    } else {
        return compute_amplitudes_double(nprojbins, nd1, nd2, nr1, nr2,
            (double *) dd, (double *) dr, (double *) rd, (double *) rr, (double *) qq, (double *) amps);
    }
}


int evaluate_xi(int nprojbins, void *amps, int nsvals, void *svals,
                      int nsbins, void *sbins, void *xi, proj_method_t proj_method, size_t element_size, char *projfn)
{
    if( ! (element_size == sizeof(float) || element_size == sizeof(double))){
        fprintf(stderr,"ERROR: In %s> Can only handle doubles or floats. Got an array of size = %zu\n",
                __FUNCTION__, element_size);
        return EXIT_FAILURE;
    }

    if(element_size == sizeof(float)) {
        return evaluate_xi_float(nprojbins, (float *) amps, nsvals, (float *) svals,
                      nsbins, (float *) sbins, (float *) xi, proj_method, projfn);
    } else {
        return evaluate_xi_double(nprojbins, (double *) amps, nsvals, (double *) svals,
                      nsbins, (double *) sbins, (double *) xi, proj_method, projfn);
    }
}

