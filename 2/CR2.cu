#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 16;    // Número predeterm. de elementos en los vectores

#include <sys/time.h>
#include <sys/resource.h>

#ifdef _noWALL_
    typedef struct rusage resnfo;
    typedef struct _timenfo {
        double time;
        double systime;
    } timenfo;
    #define timestamp(sample) getrusage(RUSAGE_SELF, (sample))
    #define printtime(t) printf("%15f s (%f user + %f sys) ",		\
            t.time + t.systime, t.time, t.systime);
#else
    typedef struct timeval resnfo;
    typedef double timenfo;
    #define timestamp(sample)     gettimeofday((sample), 0)
    #define printtime(t) printf("%15f s ", t);
#endif

void myElapsedtime(const resnfo start, const resnfo end, timenfo *const t) {
    #ifdef _noWALL_
        t->time = (end.ru_utime.tv_sec + (end.ru_utime.tv_usec * 1E-6)) 
            - (start.ru_utime.tv_sec + (start.ru_utime.tv_usec * 1E-6));
        t->systime = (end.ru_stime.tv_sec + (end.ru_stime.tv_usec * 1E-6)) 
            - (start.ru_stime.tv_sec + (start.ru_stime.tv_usec * 1E-6));
    #else
        *t = (end.tv_sec + (end.tv_usec * 1E-6)) 
            - (start.tv_sec + (start.tv_usec * 1E-6));
    #endif /*_noWALL_*/
}

/*
   Función para inicializar los vectores que vamos a utilizar
 */
void Initialization(float A[], float B[], float C[], float D[], const unsigned int n) {
    unsigned int i;

    A[0] = 0.0;
    B[0] = 2.0;
    C[0] = -1.0;
    D[0] = 1.0;

    for(i = 1; i < n - 1; i++) {
        A[i] = -1.0;
        B[i] = 2.0;
        C[i] = -1.0;
        D[i] = 0.0;
    }

    A[n - 1] = -1.0;
    B[n - 1] = 2.0;
    C[n - 1] = 0.0;
    D[n - 1] = 1.0;
}

void CR_CPU(float A[], float B[], float C[], float D[], float X[], const unsigned int n) {
    unsigned ln = floor(log2(float(n)));
    int stride, step, k, l;
    float s1, s2;

    stride = 2;
    step = 1;
    k = n - 1;

    // Forward elimination 
    for(int i = 0; i < ln - 1; i++) {
        for (int j = step; j < n - 1; j += stride) {
            s1 = A[j]/B[j-step];
            s2 = C[j]/B[j+step]; 

            A[j] = - A[j-step] * s1;
            B[j] = B[j] - C[j-step] * s1 - A[j+step] * s2;
            C[j] = -C[j+step] * s2;
            D[j] = D[j] - D[j-step]*s1-D[j+step]*s2;

        }

        // last equation
        s1 = A[k] / B[k-step];

        A[k] = - A[k-step] * s1;
        B[k] = B[k] - C[k-step] * s1;
        D[k] = D[k] - D[k-step] * s1;

        step += stride;	
        stride *= 2;
    }

    // Backward substitution
    k = n / 2 - 1;
    l = n - 1;
    s1 = (B[k] * B[l]) - (C[k]*A[l]);
    X[k] = (B[l]*D[k] - C[k]*D[l])/s1;
    X[l] = (D[l]*B[k] - D[k]*A[l])/s1;

    step = n / 4;
    stride = n / 2;
    k = step -1;

    for(int i = 0; i < ln-1; i++) {
        // First node
        X[k] = (D[k] - C[k]*X[k+step])/B[k];

        for (int j = k+stride; j < n ; j += stride) {
            X[j] = (D[j] - A[j]*X[j-step]-C[j]*X[j+step])/B[j];
        }

        step /= 2;
        stride /= 2;
        k = step -1;
    }
}

__global__ void CR_GPU(float *A, float *B, float *C, float *D, float *X, const unsigned int n) {
    extern __shared__ float s[];
    int offset = 4 * n * threadIdx.y;
    float *sA = &s[offset];
    float *sB = &s[offset + n];
    float *sC = &s[offset + 2 * n];
    float *sD = &s[offset + 3 * n];

    unsigned ln = floor(log2(float(n)));
    int stride = 2;
    int step = 1;
    int k = n - 1;
    int l = 0;
    float s1, s2;

    int idx = threadIdx.x;    // index local to Y
    int l_idx = threadIdx.x + threadIdx.y * blockDim.x;    // index local to block

    if (l_idx < n) {
        sA[idx] = A[idx];
        sB[idx] = B[idx];
        sC[idx] = C[idx];
        sD[idx] = D[idx];
    }
    __syncthreads();

    // forward elimination
    for (int i = 0; i < ln - 1; i++) {
        if (idx >= step && idx < n - 1 && idx % stride == step) {
            s1 = sA[idx] / sB[idx - step];
            s2 = sC[idx] / sB[idx + step];

            sA[idx] = - sA[idx - step] * s1;
            sB[idx] = sB[idx] - sC[idx - step] * s1 - sA[idx + step] * s2;
            sC[idx] = - sC[idx + step] * s2;
            sD[idx] = sD[idx] - sD[idx - step] * s1 - sD[idx + step] * s2;

        }

        // last equation
        if (idx == k && step <= idx) {
            s1 = sA[idx] / sB[idx - step];

            sA[idx] = - sA[idx - step] * s1;
            sB[idx] = sB[idx] - sC[idx - step] * s1;
            sD[idx] = sD[idx] - sD[idx - step] * s1;
        }

        __syncthreads();
        step += stride;
        stride *= 2;
    }

    // backward substitution
    k = n / 2 - 1;
    l = n - 1;
    s1 = (sB[k] * sB[l]) - (sC[k] * sA[l]);
    if (idx == k) { X[l_idx] = (sB[l] * sD[k] - sC[k] * sD[l]) / s1; }
    if (idx == l) { X[l_idx] = (sD[l] * sB[k] - sD[k] * sA[l]) / s1; }
    __syncthreads();

    step = n / 4;
    stride = n / 2;
    k = step - 1;

    for (int i = 0; i < ln - 1; i++) {
        if (idx == k) {
            X[l_idx] = (sD[k] - sC[k] * X[l_idx + step]) / sB[k];
        }
        __syncthreads();

        if (idx >= k && idx < n && !((idx - k) % stride)) {
            X[l_idx] = (sD[idx] - sA[idx] * X[l_idx - step] - sC[idx] * X[l_idx + step]) / sB[idx];
        }

        __syncthreads();
        step /= 2;
        stride /= 2;
        k = step - 1;
    }
}

int main(int argc, char *argv[]) {
    resnfo start, end;
    timenfo timecpu, timegpu;

    unsigned int n = (argc > 1) ? atoi(argv[1]) : N;
    unsigned int m = (argc > 2) ? atoi(argv[2]) : 1;
    unsigned int B = (1 << 24) / (n * m);
    unsigned int numBytes = n * sizeof(float);

    printf("n = %d, m = %d, nxm = %d, B = %d, shared memory = %d, blocks = %d\n",
            n, m, n * m, B, 4 * m * numBytes, (B + m - 1) / m);

    float *hA = (float *) malloc(numBytes);
    float *hB = (float *) malloc(numBytes);
    float *hC = (float *) malloc(numBytes);
    float *hD = (float *) malloc(numBytes);
    float *hX = (float *) malloc(numBytes * m);

    Initialization(hA, hB, hC, hD, n);

    float *dA, *dB, *dC, *dD, *dX;
    cudaMalloc(&dA, numBytes);
    cudaMalloc(&dB, numBytes);
    cudaMalloc(&dC, numBytes);
    cudaMalloc(&dD, numBytes);
    cudaMalloc(&dX, numBytes * m);

    cudaMemcpy(dA, hA, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, numBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dD, hD, numBytes, cudaMemcpyHostToDevice);

    dim3 dimGrid((B + m - 1) / m);
    dim3 dimBlock(n, m);

    // GPU
    timestamp(&start);
    CR_GPU<<<dimGrid, dimBlock, 4 * m * numBytes>>>(dA, dB, dC, dD, dX, n);
    cudaError_t cudaErr = cudaDeviceSynchronize();
    if (cudaErr != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaErr));
    }
    timestamp(&end);
    myElapsedtime(start, end, &timegpu);

    cudaMemcpy(hX, dX, numBytes * m, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n * m; i++) {
        // printf("X[%d] = %f\n", i, hX[i]);
        if (abs(hX[i] - 1.0) > 1.0e-6) {
            printf("Error in X[%d] = %f\n", i, hX[i]);
            break;
        }
    }

    // CPU
    timestamp(&start);
    for (int i = 0; i < B * m; i++) {
        CR_CPU(hA, hB, hC, hD, hX, n);
    }
    timestamp(&end);
    myElapsedtime(start, end, &timecpu);

    printtime(timecpu);
    printf(" -> CR in CPU\n");
    printtime(timegpu);
    printf(" -> CR in GPU\n");
    printf("Speedup = %f\n", timecpu / timegpu);

    free(hA);
    free(hB);
    free(hC);
    free(hD);
    free(hX);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dD);
    cudaFree(dX);
}
