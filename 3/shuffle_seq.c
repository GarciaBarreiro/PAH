#include <stdio.h>
#include <stdlib.h>
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

float getMin(float *A, int n, int p, int i, int j) {
    float min = A[i * n * p + j * p];
    for (int k = 1; k < p; k++) {
        if (A[i * n * p + j * p + k] < min) {
            min = A[i * n * p + j * p + k];
        }
    }
    return min;
}

int main(int argc, char *argv[]) {
    struct timeval t_prev, t_init, t_final;
    double seq_t;

    if (argc < 4) {
        printf("Usage: %s M N P\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int P = atoi(argv[3]);

    float *X = malloc(M * N * P * sizeof(float));

    if (!X || !M || !N || !P) {
        printf("Memory allocation failed\n");
        return 1;
    }

    _initMatrix(X, M, N, P);

    printf("INPUT:\n");
    _printMatrix(X, M, N, P);

    INIT_TIME(t_prev, t_init);
    // step 1
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float min = getMin(X, N, P, i, j);
            for (int k = 0; k < P; k++) {
                X[i * N * P + j * P + k] += min;
            }
        }
    }

    if (DEBUG) {
        printf("STEP 1:\n");
        _printMatrix(X, M, N, P);
    }

    float *t = malloc(M * N * P * sizeof(float));

    // step 2
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                t[i * N * P + j * P + k] = X[i * N * P + j * P + k];
                if (j) {
                    t[i * N * P + j * P + k] += X[i * N * P + (j - 1) * P + k];
                }
                if (j < N - 1) {
                    t[i * N * P + j * P + k] += X[i * N * P + (j + 1) * P + k];
                }
            }
        }
    }

    if (DEBUG) {
        printf("STEP 2:\n");
        _printMatrix(t, M, N, P);
    }

    // step 3
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < P; k++) {
                X[i * N * P + j * P + k] = t[i * N * P + j * P + k];
                if (i) {
                    X[i * N * P + j * P + k] += t[(i - 1) * N * P + j * P + k];
                }
                if (i < M - 1) {
                    X[i * N * P + j * P + k] += t[(i + 1) * N * P + j * P + k];
                }
            }
        }
    }
    GET_TIME(t_prev, t_init, t_final, seq_t);

    printf("OUTPUT:\n");
    _printMatrix(X, M, N, P);

    free(X);
    free(t);

    FILE *fp = fopen((argc > 4) ? argv[4] : "seq_out.csv", "a");
    fprintf(fp, "%d,%d,%d,%ld,%f,%d\n", M, N, P, N * P * sizeof(float), seq_t, N * P);
    fclose(fp);

    return 0;
}
