#pragma once
#include <cublas_v2.h>

// Transposes d_in of shape (d0, d1, d2, d3) to d_out of shape (d0, d1, d3, d2).
// Uses cublasSgeam with batch count = d0 * d1, transposing each (d2 x d3) sub-matrix.
void transpose_cublas(cublasHandle_t handle,
                      const double*   d_in,
                      double*         d_out,
                      int d0, int d1, int d2, int d3);
