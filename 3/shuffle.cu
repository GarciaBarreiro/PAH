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

__global__ void kernel1(float *A, int m, int n, int p) {
    extern __shared__ float s[];
    int m_idx = blockIdx.x;
    int t_x = threadIdx.x;
    int t_y = threadIdx.y;

    s[t_x * p + t_y] = A[m_idx * n * p + t_x * p + t_y];

    // STEP 1
    float lmin = s[t_x * p + t_y];
    for (int offset = p / 2; offset > 0; offset /= 2) {
        lmin = min(lmin, __shfl_down_sync(0xFFFFFFFF, lmin, offset, p));
    }

    // TODO: try to use __shfl_sync instead of going through shared memory
    // lmin = __shfl_sync(0xFFFFFFFF, lmin, 0);
    lmin = s[t_x * p];

    s[t_x * p + t_y] += lmin;

    // STEP 2
    if (t_x == 0) {
        A[m_idx * n * p + t_x * p + t_y] = s[t_x * p + t_y] + s[(t_x + 1) * p + t_y];
    } else if (t_x == n - 1) {
        A[m_idx * n * p + t_x * p + t_y] = s[t_x * p + t_y] + s[(t_x - 1) * p + t_y];
    } else {
        A[m_idx * n * p + t_x * p + t_y] = s[t_x * p + t_y] + s[(t_x - 1) * p + t_y] + s[(t_x + 1) * p + t_y];
    }
}

__global__ void kernel2() {
}

int main(int argc, char *argv[]) {
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

    float *dX;
    cudaMalloc(&dX, numBytes);
    cudaMemcpy(dX, X, numBytes, cudaMemcpyHostToDevice);

    dim3 dimGrid(M);
    dim3 dimBlock(N, P);
    
    kernel1<<<dimGrid, dimBlock, N * P * sizeof(float)>>>(dX, M, N, P);
    SYNC;

    cudaMemcpy(X, dX, numBytes, cudaMemcpyDeviceToHost);

    printf("OUTPUT:\n");
    _printMatrix(X, M, N, P);

    return 0;
}
