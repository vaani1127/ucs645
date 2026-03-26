#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TOTAL_SIZE 5000000   // 500 million elements

int main(int argc, char *argv[])
{
    int rank, size;
    double multiplier;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Step 1: Input multiplier from user */
    if (rank == 0)
    {
        scanf("%lf", &multiplier);
    }

    /* Step 2: Broadcast multiplier */
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Step 3: Determine local chunk size */
    long local_size = TOTAL_SIZE / size;

    double *A = (double *)malloc(local_size * sizeof(double));
    double *B = (double *)malloc(local_size * sizeof(double));

    /* Initialize vectors locally */
    for (long i = 0; i < local_size; i++)
    {
        A[i] = 1.0;
        B[i] = 2.0 * multiplier;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    double start = MPI_Wtime();

    /* Step 4: Local computation */
    double local_sum = 0.0;

    for (long i = 0; i < local_size; i++)
    {
        local_sum += A[i] * B[i];
    }

    /* Step 5: Global reduction */
    double final_result = 0.0;

    MPI_Reduce(&local_sum,
               &final_result,
               1,
               MPI_DOUBLE,
               MPI_SUM,
               0,
               MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0)
    {
        printf("Final Dot Product: %f\n", final_result);
        printf("Total Execution Time: %f seconds\n", end - start);
    }

    free(A);
    free(B);

    MPI_Finalize();

}