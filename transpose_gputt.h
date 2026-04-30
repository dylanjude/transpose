#pragma once
#include <gputt.h>

// Transposes d_in of shape (d0, d1, d2, d3) to d_out of shape (d0, d1, d3, d2).
// plan must already be created by the caller (no allocation done here).
void transpose_gputt(gputtHandle plan, const double* d_in, double* d_out);
