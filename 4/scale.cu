#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sys/time.h>

#ifndef DEBUG
    #define DEBUG 0
#else
    #define DEBUG 1
#endif

#define INIT_TIME(prev, init) \
    gettimeofday(&prev, NULL); \
    gettimeofday(&init, NULL);

// remove overhead created by call to gettimeofday
#define GET_TIME(prev, init, final, res) \
    gettimeofday(&final, NULL); \
    res = (final.tv_sec-init.tv_sec+(final.tv_usec-init.tv_usec)/1.e6) - \
          (init.tv_sec-prev.tv_sec+(init.tv_usec-prev.tv_usec)/1.e6);

void _printMat(float *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.3f ", A[i * n + j]);
        }
        printf("\n");
    }
}

void _printVec(float *v, int n) {
    printf("[");
    for (int i = 0; i < n; i++) {
        printf("%.2f ", v[i]);
    }
    printf("\b]\n");
}

__global__ void factorMat(float *A, int m, int n, float factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < m * n) {
        A[i] *= factor;
    }
}

int main(int argc, char **argv) {
    struct timeval t_prev, t_init, t_final;
    double htran_t, dtran_t, kernel_t;  // host to device, device to host, and kernel time

    if (argc < 5) {
        printf("Usage: %s <m> <n> <rep> <factors>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int rep = atoi(argv[3]);
    char *f = argv[4]; // temp var
    float *factor = (float *)malloc(rep * sizeof(float));
    int thr_block = (argc > 5) ? atoi(argv[5]) : 32;      // threads per block

    // checks everything's OK, splits and saves to factor
    if (!(f[0] == '[' && f[strlen(f) - 1] == ']')) {
        printf("Factors are inputted like [0.1,0.2,0.6]\n");
        return 1;
    }

    // converts factor string to an array of floats
    f[strlen(f) - 1] = '\0';
    if (rep == 1) {
        factor[0] = atof(++f);
    } else {
        char *fact = strtok(++f, ",");
        for (int i = 0; i < rep; i++) {
            if (!fact) {
                printf("Not enough factors\n");
                return 1;
            }
            factor[i] = atof(fact);
            fact = strtok(NULL, ",");
        }
    }

    if (DEBUG) _printVec(factor, rep);

    unsigned int numBytes = m * n * sizeof(float);

    // init host
    float *cA = (float *)malloc(numBytes);  // in
    for (int i = 0; i < m * n; i++) { cA[i] = i; }
    float *cB = (float *)malloc(numBytes);  // out

    // init device
    float *A;
    cudaMalloc(&A, numBytes);

    INIT_TIME(t_prev, t_init);

    cudaMemcpy(A, cA, numBytes, cudaMemcpyHostToDevice);

    GET_TIME(t_prev, t_init, t_final, htran_t);

    dim3 dimBlock(thr_block);
    dim3 dimGrid((n * m + dimBlock.x - 1) / dimBlock.x);

    cudaStream_t *streams = (cudaStream_t *)malloc(rep * sizeof(cudaStream_t));

    for (int i = 0; i < rep; i++) {
        cudaStreamCreate(&streams[i]);
    }

    if (DEBUG) {
        printf("A:\n");
        _printMat(cA, m, n);
        printf("\n");
    }

    INIT_TIME(t_prev, t_init);

    for (int i = 0; i < rep; i++) {
        factorMat<<<dimGrid, dimBlock, 0, streams[i]>>>(A, m, n, factor[i]);
        /*
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != cudaSuccess) {
            printf("Kernel launch failed: %s\n",  cudaGetErrorString(cudaerr));
        }
        */
        cudaMemcpyAsync(cB, A, numBytes, cudaMemcpyDeviceToHost, streams[i]);
        if (DEBUG) {
            printf("%d >>>>>>>>>>>>>>>>>>> %.2f\n", i, factor[i]);
            _printMat(cB, m, n);
            printf("\n");
        }
    }

    GET_TIME(t_prev, t_init, t_final, kernel_t);

    INIT_TIME(t_prev, t_init);

    for (int i = 0; i < rep; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // cudaMemcpy(cB, A, numBytes, cudaMemcpyDeviceToHost);

    GET_TIME(t_prev, t_init, t_final, dtran_t);

    /*
    if (DEBUG) {
        printf("A:\n");
        _printMat(cA, m, n);

        printf("B:\n");
        _printMat(cB, m, n);
    }
    */

    cudaFree(A);

    free(factor);
    free(cA);
    free(cB);
    free(streams);

    // write results
    FILE *fp = fopen((argc > 6) ? argv[6] : "out.csv", "a");
    if (!fp) { printf("Error opening file\n"); }
    else {
        fprintf(fp, "%d,%d,%d,%d,%f,%f,%f\n", m, n, thr_block, rep, htran_t + dtran_t, kernel_t, htran_t + dtran_t + kernel_t);
        fclose(fp);
    }

    if (DEBUG) {
        printf("Host transfer time: %f\n", htran_t);
        printf("Device transfer time: %f\n", dtran_t);
        printf("Kernel time: %f\n", kernel_t);
        printf("Total time: %f\n", htran_t + kernel_t);
    }

    return 0;
}
