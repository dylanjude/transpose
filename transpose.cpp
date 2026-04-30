#include "transpose.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CHECK_CUBLAS(call)                                                      \
    do {                                                                        \
        cublasStatus_t _s = (call);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d: status %d\n",                 \
                    __FILE__, __LINE__, (int)_s);                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

void transpose_cublas(cublasHandle_t handle,
               const double*   d_in,   // row-major (d0, d1, d2, d3)
               double*         d_out,  // row-major (d0, d1, d3, d2)
               int d0, int d1, int d2, int d3)
{
    int       batchCount = d0 * d1;
    long long stride     = (long long)d2 * d3;

    const double alpha = 1.0, beta = 0.0;

    // cuBLAS is column-major. Interpret each row-major (d2 x d3) matrix
    // as a column-major (d3 x d2) matrix (lda = d3). The transposed
    // row-major (d3 x d2) output maps to column-major (d2 x d3) (ldc = d2).
    //
    // cublasSgeam:  C(m x n) = alpha * op(A) + beta * op(B)
    //   m = d2, n = d3
    //   transa = CUBLAS_OP_T  =>  A stored as (n x m) col-major, lda = d3
    //   C stored as (m x n) col-major, ldc = d2
    //   beta = 0  =>  B is not referenced (pass nullptr)
    //
    // cublasSgeamBatched does not exist in cuBLAS; loop over individual calls.
    for (int b = 0; b < batchCount; ++b) {
        const double* A = d_in  + b * stride;
        double*       C = d_out + b * stride;
        CHECK_CUBLAS(cublasDgeam(
            handle,
            CUBLAS_OP_T,   // transpose input
            CUBLAS_OP_N,   // op(B): irrelevant (beta = 0)
            d2, d3,        // m, n
            &alpha,
            A,  d3,        // A: col-major (d3 x d2), lda = d3
            &beta,
            nullptr, d2,   // B: not referenced
            C,  d2));      // C: col-major (d2 x d3), ldc = d2
    }
}
