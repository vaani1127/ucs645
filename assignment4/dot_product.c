#include <mpi.h>
#include <stdio.h>

#define N 8

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A[N] = {1,2,3,4,5,6,7,8};
    int B[N] = {8,7,6,5,4,3,2,1};

    int local_size = N / size;
    int local_A[local_size], local_B[local_size];

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    MPI_Scatter(A, local_size, MPI_INT, local_A, local_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, local_size, MPI_INT, local_B, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_dot = 0;
    for (int i = 0; i < local_size; i++)
        local_dot += local_A[i] * local_B[i];

    int global_dot;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Dot Product = %d\n", global_dot);
        printf("Execution Time = %f seconds\n\n", end - start);
    }

    MPI_Finalize();
    return 0;
}