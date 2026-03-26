#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1000000   // safe size

void MyBcast(double *buffer, int count, int rank, int size)
{
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            MPI_Send(buffer, count, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Recv(buffer, count, MPI_DOUBLE, 0, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

int main(int argc, char *argv[])
{
    int rank, size;
    double *buffer;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    buffer = (double *)malloc(ARRAY_SIZE * sizeof(double));

    if (buffer == NULL)
    {
        printf("Memory allocation failed\n");
        MPI_Finalize();
        return 0;
    }

    /* Initialize data */
    if (rank == 0)
    {
        for (int i = 0; i < ARRAY_SIZE; i++)
            buffer[i] = 1.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Custom Broadcast */
    double start1 = MPI_Wtime();

    MyBcast(buffer, ARRAY_SIZE, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);

    double end1 = MPI_Wtime();

    /* Reset data */
    if (rank == 0)
    {
        for (int i = 0; i < ARRAY_SIZE; i++)
            buffer[i] = 2.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* MPI Broadcast */
    double start2 = MPI_Wtime();

    MPI_Bcast(buffer, ARRAY_SIZE,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    double end2 = MPI_Wtime();

    if (rank == 0)
    {
        printf("Processes: %d\n", size);
        printf("MyBcast Time: %f seconds\n", end1 - start1);
        printf("MPI_Bcast Time: %f seconds\n", end2 - start2);
    }

    free(buffer);

    MPI_Finalize();
    return 0;
}