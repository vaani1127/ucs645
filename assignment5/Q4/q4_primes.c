#include <mpi.h>
#include <stdio.h>
#include <math.h>

int is_prime(int n)
{
    if (n < 2) return 0;
    for (int i = 2; i <= sqrt(n); i++)
    {
        if (n % i == 0)
            return 0;
    }
    return 1;
}
int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int number;
    int max_value = 100; // You can change this

    MPI_Status status;

    if (rank == 0)
    {
        int next_number = 2;
        int active_slaves = size - 1;

        while (active_slaves > 0)
        {
            MPI_Recv(&number,
                     1,
                     MPI_INT,
                     MPI_ANY_SOURCE,
                     0,
                     MPI_COMM_WORLD,
                     &status);

            int slave = status.MPI_SOURCE;

            if (number > 0)
                printf("Prime found: %d\n", number);

            if (next_number <= max_value)
            {
                MPI_Send(&next_number,
                         1,
                         MPI_INT,
                         slave,
                         0,
                         MPI_COMM_WORLD);

                next_number++;
            }
            else
            {
                int stop = 0;

                MPI_Send(&stop,
                         1,
                         MPI_INT,
                         slave,
                         0,
                         MPI_COMM_WORLD);

                active_slaves--;
            }
        }
    }
    else
    {
        number = 0;

        while (1)
        {
            MPI_Send(&number,
                     1,
                     MPI_INT,
                     0,
                     0,
                     MPI_COMM_WORLD);

            MPI_Recv(&number,
                     1,
                     MPI_INT,
                     0,
                     0,
                     MPI_COMM_WORLD,
                     &status);

            if (number == 0)
                break;

            if (is_prime(number))
                number = number;
            else
                number = -number;
        }
    }

    MPI_Finalize();

    return 0;
}