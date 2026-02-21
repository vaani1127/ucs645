#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    srand(time(NULL) + rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    int local_max = 0, local_min = 1000;
    for (int i = 0; i < 10; i++) {
        int num = rand() % 1001;
        if (num > local_max) local_max = num;
        if (num < local_min) local_min = num;
    }

    struct { int value; int rank; } max_pair, min_pair, global_max, global_min;

    max_pair.value = local_max;
    max_pair.rank = rank;

    min_pair.value = local_min;
    min_pair.rank = rank;

    MPI_Reduce(&max_pair, &global_max, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(&min_pair, &global_min, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Global Maximum = %d (Process %d)\n", global_max.value, global_max.rank);
        printf("Global Minimum = %d (Process %d)\n", global_min.value, global_min.rank);
        printf("Execution Time = %f seconds\n\n", end - start);
    }

    MPI_Finalize();
    return 0;
}