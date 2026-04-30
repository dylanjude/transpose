#include "transpose_gputt.h"
#include <gputt.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_GPUTT(call)                                                       \
    do {                                                                        \
        gputtResult _r = (call);                                                \
        if (_r != GPUTT_SUCCESS) {                                              \
            fprintf(stderr, "gpuTT error %s:%d: code %d\n",                    \
                    __FILE__, __LINE__, (int)_r);                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

void transpose_gputt(gputtHandle plan, const float* d_in, float* d_out)
{
    CHECK_GPUTT(gputtExecute(plan, d_in, d_out));
}
