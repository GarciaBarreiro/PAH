#include "2D_conv.h"

void func(const dt w[K], const dt  data_IN[N][N], dt data_OUT[N][N]) {
    // cache kernel, eliminating memory access on PIPE_LOOP
    const dt w0 = w[0], w1 = w[1], w2 = w[2];   // top row
    const dt w3 = w[3], w4 = w[4], w5 = w[5];   // middle row
    const dt w6 = w[6], w7 = w[7], w8 = w[8];   // bottom row

    // save the top row of data_IN locally
    dt top_row[N];
    for (int i = 0; i < N; ++i) {
        top_row[i] = data_IN[0][i];
    }
    // Ignore boundaries conditions
    for (int i = 1; i < N - 1; ++i) {
        // init the pointers 3x3 window
        dt d00 = top_row[0], d01 = top_row[1], d02;
        dt d10 = data_IN[i][0], d11 = data_IN[i][1], d12;
        dt d20 = data_IN[i+1][0], d21 = data_IN[i+1][1], d22;
        PIPE_LOOP: for (int j = 1; j < N - 1; ++j) {
            #pragma HLS PIPELINE II=1
            // get new column. because we have top_row locally
            // we only load from memory d12 and d22
            d02 = top_row[j+1];
            d12 = data_IN[i][j+1];
            d22 = data_IN[i+1][j+1];

            // convolution
            data_OUT[i][j] = w0 * d00 + w1 * d01 + w2 * d02 +
                w3 * d10 + w4 * d11 + w5 * d12 +
                w6 * d20 + w7 * d21 + w8 * d22;

            // after calculating the convolution, update the pointers
            d00 = d01;
            d01 = d02;
            top_row[i-1] = d10; // update the top row with the middle one
            d10 = d11;
            d11 = d12;
            d20 = d21;
            d21 = d22;
        }
        // last two elements of the top row
        top_row[N-2] = d10; // after last iteration, d10 holds d11 (N-2)
        top_row[N-1] = d11; // and d11 holds d12 (N-1)
    }
}
