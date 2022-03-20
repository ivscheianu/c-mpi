#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mpi.h"

// #define DEBUG

int MPI_Bucket_sort(int array_size, double max, double* array, int root, MPI_Comm comm);
int comparator(const void* a, const void* b);

const double MAX = 9999.9;
const int NUMBER_OF_ELEMENTS = 1000000;

int main(int argc, char* argv[])
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	double* array = calloc(NUMBER_OF_ELEMENTS, sizeof(double));

	if (rank == 0)
	{
		srand(((int)time(NULL) + rank));
		for (int index = 0; index < NUMBER_OF_ELEMENTS; index++)
		{
			array[index] = ((double)rand() / RAND_MAX) * MAX;
		}
	}

#ifdef DEBUG
	{
		printf("Unsorted: \n");
		if (rank == 0)
		{
			for (int index = 0; index < NUMBER_OF_ELEMENTS; index++)
			{
				printf("%lf \n", array[index]);
			}
		}
	}
#endif

	const double start_time = MPI_Wtime();
	MPI_Bucket_sort(NUMBER_OF_ELEMENTS, MAX, array, 0, MPI_COMM_WORLD);
	printf("All processing took %lf\n", MPI_Wtime() - start_time);

#ifdef DEBUG
	{
		printf("Sorted: \n");
		if (rank == 0)
		{
			for (int index = 0; index < NUMBER_OF_ELEMENTS; index++)
			{
				printf("%lf \n", array[index]);
			}
		}
	}
#endif
	
	MPI_Finalize();
}

int MPI_Bucket_sort(const int array_size, const double max, double* array, const int root, const MPI_Comm comm)
{
	double communication_time = 0.0, computation_time = 0.0;
	int size, rank, counter = 0;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	int* counters = calloc(size, sizeof(int));
	int* displacement = calloc(size, sizeof(int));

	double* bucket = calloc(array_size, sizeof(double));

	double time = MPI_Wtime();
	MPI_Bcast(&array[0], array_size, MPI_DOUBLE, root, comm);
	time = MPI_Wtime() - time;
	communication_time += time;

	time = MPI_Wtime();
	for (int i = 0; i < array_size; i++)
	{
		if (rank * max / size <= array[i] && array[i] < (rank + 1) * max / size)
		{
			bucket[counter++] = array[i];
		}
	}
	time = MPI_Wtime() - time;
	computation_time += time;

	time = MPI_Wtime();
	qsort(bucket, counter, sizeof(double), comparator);
	time = MPI_Wtime() - time;
	computation_time += time;

	time = MPI_Wtime();
	MPI_Gather(&counter, 1, MPI_INT, counters, 1, MPI_INT, root, comm);
	time = MPI_Wtime() - time;
	communication_time += time;

	time = MPI_Wtime();
	displacement[0] = 0;
	for (int index = 0; index < size - 1; index++)
	{
		displacement[index + 1] = displacement[index] + counters[index];
	}
	time = MPI_Wtime() - time;
	computation_time += time;

	time = MPI_Wtime();
	MPI_Gatherv(bucket, counter, MPI_DOUBLE, array, counters, displacement, MPI_DOUBLE, root, comm);
	time = MPI_Wtime() - time;
	communication_time += time;

	printf("Processor number %d:\n-computation time: %lf\n-communication = %lf\n-total execution = %lf \n", rank, computation_time, communication_time, computation_time + communication_time);

	return MPI_SUCCESS;
}

int comparator(const void* a, const void* b)
{
	if (*(double*)a > *(double*)b) {
		return 1;
	}
	if (*(double*)a < *(double*)b)
	{
		return -1;
	}
	return 0;
}
