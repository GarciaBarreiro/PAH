#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define INIT_TIME(prev, init) \
    gettimeofday(&prev, NULL); \
    gettimeofday(&init, NULL);

// remove overhead created by call to gettimeofday
#define GET_TIME(prev, init, final, res) \
    gettimeofday(&final, NULL); \
    res = (final.tv_sec-init.tv_sec+(final.tv_usec-init.tv_usec)/1.e6) - \
          (init.tv_sec-prev.tv_sec+(init.tv_usec-prev.tv_usec)/1.e6);

#define SYNC \
    { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err)); \
        } \
    }

void _printMatrix(float *A, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++){
                printf("%.2f ", A[i * n * p + j * p + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void _initMatrix(float *A, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < p; k++) {
                A[i * n * p + j * p + k] = k + 1.0;
            }
        }
    }
}

__global__ void kernel1(float *A, int n, int p) {
    extern __shared__ float s[];
    int b_idx = blockIdx.x;
    int i_p = threadIdx.x;  // iterates over p
    int i_n = threadIdx.y;  // iterates over n (cuda stores matrices by column)

    s[i_n * p + i_p] = A[b_idx * n * p + i_n * p + i_p];

    // STEP 1
    float l_min = s[i_n * p + i_p];
    for (int offset = p / 2; offset > 0; offset /= 2) {
        l_min = min(l_min, __shfl_down_sync(0xFFFFFFFF, l_min, offset, p));
    }

    l_min = __shfl_sync(0xFFFFFFFF, l_min, 0, p);
    s[i_n * p + i_p] += l_min;

    // sync because we need to make sure all threads have updated their values
    // before entering the next step
    __syncthreads();

    // STEP 2
    if (i_n == 0) {
        A[b_idx * n * p + i_n * p + i_p] = s[i_n * p + i_p] + s[(i_n + 1) * p + i_p];
    } else if (i_n == n - 1) {
        A[b_idx * n * p + i_n * p + i_p] = s[i_n * p + i_p] + s[(i_n - 1) * p + i_p];
    } else {
        A[b_idx * n * p + i_n * p + i_p] = s[i_n * p + i_p] + s[(i_n - 1) * p + i_p] + s[(i_n + 1) * p + i_p];
    }
}

// use a second matrix instead of the same one for input and output,
// this way we can avoid barriers inside the kernel
__global__ void kernel2(float *A, float *B, int m, int n, int p) {
    int m_idx = blockIdx.x;
    int t_idx = threadIdx.y * p + threadIdx.x;

    // STEP 3
    float l_val = A[m_idx * n * p + t_idx];
    if (m_idx == 0) {
        l_val += A[(m_idx + 1) * n * p + t_idx];
    } else if (m_idx == m - 1) {
        l_val += A[(m_idx - 1) * n * p + t_idx];
    } else {
        l_val += A[(m_idx - 1) * n * p + t_idx] + A[(m_idx + 1) * n * p + t_idx];
    }
    B[m_idx * n * p + t_idx] = l_val;
}

int main(int argc, char *argv[]) {
    struct timeval t_prev, t_init, t_final;
    double kernel_t;

    if (argc < 4) {
        printf("Usage: %s M N P\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int P = atoi(argv[3]);

    unsigned int numBytes = M * N * P * sizeof(float);
    float *X = (float *)malloc(numBytes);

    if (!X || !numBytes) {
        printf("Memory allocation failed\n");
        return 1;
    }

    _initMatrix(X, M, N, P);

    printf("INPUT:\n");
    _printMatrix(X, M, N, P);

    float *dX, *dY;
    cudaMalloc(&dX, numBytes);
    cudaMemcpy(dX, X, numBytes, cudaMemcpyHostToDevice);
    cudaMalloc(&dY, numBytes);

    dim3 dimGrid(M);
    dim3 dimBlock(P, N);
    
    INIT_TIME(t_prev, t_init);
    kernel1<<<dimGrid, dimBlock, N * P * sizeof(float)>>>(dX, N, P);
    SYNC;
    kernel2<<<dimGrid, dimBlock>>>(dX, dY, M, N, P);
    SYNC;
    GET_TIME(t_prev, t_init, t_final, kernel_t);

    cudaMemcpy(X, dY, numBytes, cudaMemcpyDeviceToHost);

    printf("OUTPUT:\n");
    _printMatrix(X, M, N, P);

    cudaFree(dX);
    cudaFree(dY);
    free(X);

    FILE *fp = fopen((argc > 4) ? argv[4] : "out.csv", "a");
    fprintf(fp, "%d,%d,%d,%ld,%f,%d\n", M, N, P, N * P * sizeof(float), kernel_t, N * P);
    fclose(fp);

    return 0;
}
