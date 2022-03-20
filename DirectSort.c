#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

// #define DEBUG

const double MAX = 9999.9;
const int NUMBER_OF_ELEMENTS = 1000000;

int MPI_Direct_sort(int array_size, double* array, int root, MPI_Comm comm);
double* merge_array(int first_array_size, const double* first_array, int second_array_size, const double* second_array);
void merge_sort(int array_size, double* array);
void swap(double* a, double* b);

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

		for (int i = 0; i < NUMBER_OF_ELEMENTS; i++)
		{
			array[i] = ((double)rand() / RAND_MAX) * MAX;
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
	MPI_Direct_sort(NUMBER_OF_ELEMENTS, array, 0, MPI_COMM_WORLD);
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


int MPI_Direct_sort(const int array_size, double* array, const int root, const MPI_Comm comm)
{
	double communication_time = 0.0, computation_time = 0.0;
	int rank, size;

	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	double* local_a = calloc(array_size / size, sizeof(double));
	double time = MPI_Wtime();
	MPI_Scatter(array, array_size / size, MPI_DOUBLE, local_a, array_size / size, MPI_DOUBLE, root, comm);
	time = MPI_Wtime() - time;
	communication_time += time;

	time = MPI_Wtime();
	merge_sort(array_size / size, local_a);
	time = MPI_Wtime() - time;
	computation_time += time;

	time = MPI_Wtime();
	MPI_Gather(local_a, array_size / size, MPI_DOUBLE, array, array_size / size, MPI_DOUBLE, root, comm);
	time = MPI_Wtime() - time;
	communication_time += time;

	time = MPI_Wtime();
	if (rank == 0)
	{
		for (int index = 1; index < size; index++)
		{
			double* temp = merge_array(index * array_size / size, array, array_size / size, array + index * array_size / size);
			for (int index2 = 0; index2 < (index + 1) * array_size / size; index2++)
			{
				array[index2] = temp[index2];
			}
		}
	}
	time = MPI_Wtime() - time;
	computation_time += time;

	printf("Processor number %d:\n-computation time: %lf\n-communication = %lf\n-total execution = %lf \n", rank, computation_time, communication_time, computation_time + communication_time);
	return MPI_SUCCESS;
}

double* merge_array(const int first_array_size, const double* first_array, const int second_array_size, const double* second_array)
{
	int i, j, k;
	double* merged_array = calloc(first_array_size + second_array_size, sizeof(double));

	for (i = j = k = 0; i < first_array_size && j < second_array_size;)
	{
		if (first_array[i] <= second_array[j])
		{
			merged_array[k++] = first_array[i++];
		}
		else
		{
			merged_array[k++] = second_array[j++];
		}
	}

	if (i == first_array_size)
	{
		while (j < second_array_size)
		{
			merged_array[k++] = second_array[j++];
		}
	}
	else
	{
		while (i < first_array_size)
		{
			merged_array[k++] = first_array[i++];
		}
	}

	return merged_array;
}


void merge_sort(const int array_size, double* array)
{
	if (array_size <= 1)
	{
		return;
	}
	if (array_size == 2)
	{
		if (array[0] > array[1])
		{
			swap(&array[0], &array[1]);
		}
		return;
	}
	merge_sort(array_size / 2, array);
	merge_sort(array_size - array_size / 2, array + array_size / 2);
	double* merged_array = merge_array(array_size / 2, array, array_size - array_size / 2, array + array_size / 2);
	for (int i = 0; i < array_size; i++)
	{
		array[i] = merged_array[i];
	}
}

void swap(double* a, double* b)
{
	const double temp = *a;
	*a = *b;
	*b = temp;
}
