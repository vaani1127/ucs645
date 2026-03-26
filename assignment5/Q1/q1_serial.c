#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 65536   // 2^16

int main() {
    double *X, *Y;
    double a = 2.5;

    X = (double*)malloc(N * sizeof(double));
    Y = (double*)malloc(N * sizeof(double));

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        X[i] = 1.0;
        Y[i] = 2.0;
    }

    clock_t start = clock();

    // DAXPY computation
    for (int i = 0; i < N; i++) {
        X[i] = a * X[i] + Y[i];
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Serial Execution Time: %f seconds\n", time_taken);

    free(X);
    free(Y);

    return 0;
}