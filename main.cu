#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <gputt.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "transpose.h"
#include "transpose_gputt.h"

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

#define CHECK_GPUTT(call)                                                       \
    do {                                                                        \
        gputtResult _r = (call);                                                \
        if (_r != GPUTT_SUCCESS) {                                              \
            fprintf(stderr, "gpuTT error %s:%d: code %d\n",                    \
                    __FILE__, __LINE__, (int)_r);                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// Returns elapsed milliseconds between two already-recorded/synchronized events.
static float elapsed_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static void report(const char* label, float ms, long long n_elems)
{
    double bytes     = 2.0 * n_elems * sizeof(double);  // read + write
    double bandwidth = bytes / (ms * 1e-3) / 1e9;
    printf("%-14s  %8.3f ms   %6.1f GB/s\n", label, ms, bandwidth);
}

// Fill: data[i] = (i % 10000) * 1e-4  (deterministic values in [0, 1))
__global__ void fill_kernel(double* data, long long n)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        data[idx] = (double)(idx % 10000) * 1e-4;
}

int main()
{
    // ---- dimensions ----
    const int D0 = 6, D1 = 100, D2 = 32, D3 = 10648;
    const long long N = (long long)D0 * D1 * D2 * D3;

    printf("Input  shape: (%d, %d, %d, %d)  — %.1f MB\n",
           D0, D1, D2, D3, N * sizeof(double) / 1048576.0);
    printf("Output shape: (%d, %d, %d, %d)  — %.1f MB\n\n",
           D0, D1, D3, D2, N * sizeof(double) / 1048576.0);

    // ---- allocate device buffers ----
    double *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in,  N * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(double)));

    // ---- fill input ----
    const int THREADS = 256;
    long long  BLOCKS  = (N + THREADS - 1) / THREADS;
    fill_kernel<<<BLOCKS, THREADS>>>(d_in, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // ---- cuBLAS handle ----
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));

    // ---- gpuTT plan ----
    // gputt uses column-major convention (fastest index first).
    // Row-major (D0, D1, D2, D3) == column-major (D3, D2, D1, D0).
    // We want to swap the last two row-major dims (D2 <-> D3),
    // which is swapping the first two column-major dims: permutation = {1,0,2,3}.
    gputtHandle gputt_plan;
    int gputt_dim[4]  = {D3, D2, D1, D0};
    int gputt_perm[4] = {1, 0, 2, 3};
    CHECK_GPUTT(gputtPlan(&gputt_plan, 4, gputt_dim, gputt_perm,
                          gputtDataTypeFloat64, /*stream=*/0));

    // ---- CUDA events for timing ----
    cudaEvent_t ev0, ev1;
    CHECK_CUDA(cudaEventCreate(&ev0));
    CHECK_CUDA(cudaEventCreate(&ev1));

    // ---- transpose_cublas ----
    CHECK_CUDA(cudaEventRecord(ev0));
    transpose_cublas(cublas_handle, d_in, d_out, D0, D1, D2, D3);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    report("cublas", elapsed_ms(ev0, ev1), N);

    // ---- transpose_gputt ----
    CHECK_CUDA(cudaEventRecord(ev0));
    transpose_gputt(gputt_plan, d_in, d_out);
    CHECK_CUDA(cudaEventRecord(ev1));
    CHECK_CUDA(cudaEventSynchronize(ev1));
    report("gputt", elapsed_ms(ev0, ev1), N);

    // ---- verify first batch element against cublas reference ----
    // Reuse d_out: run cublas first, snapshot the slice, then run gputt and compare.
    const int VERIFY_ELEMS = D2 * D3;
    std::vector<double> h_ref(VERIFY_ELEMS), h_out(VERIFY_ELEMS);

    transpose_cublas(cublas_handle, d_in, d_out, D0, D1, D2, D3);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_out, VERIFY_ELEMS * sizeof(double), cudaMemcpyDeviceToHost));

    transpose_gputt(gputt_plan, d_in, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, VERIFY_ELEMS * sizeof(double), cudaMemcpyDeviceToHost));

    bool ok = true;
    int  errors = 0;
    for (int j = 0; j < D3 && errors < 10; ++j) {
        for (int i = 0; i < D2 && errors < 10; ++i) {
            double ref = h_ref[j * D2 + i];
            double got = h_out[j * D2 + i];
            if (fabs(ref - got) > 1e-12) {
                printf("  MISMATCH [0,0,%d,%d]: cublas=%.6f  gputt=%.6f\n", i, j, ref, got);
                ok = false;
                ++errors;
            }
        }
    }
    printf("\nVerification vs cublas (batch [0,0]): %s\n", ok ? "PASSED" : "FAILED");

    // ---- cleanup ----
    CHECK_CUDA(cudaEventDestroy(ev0));
    CHECK_CUDA(cudaEventDestroy(ev1));
    CHECK_GPUTT(gputtDestroy(gputt_plan));
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return ok ? 0 : 1;
}
