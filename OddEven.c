#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

// #define DEBUG

const double MAX = 9999.9;
const int NUMBER_OF_ELEMENTS = 1000000;

int MPI_OddEven_sort(int array_size, double* first_array, int root, MPI_Comm comm);
void exchange(int array_size, double* first_array, int sender, int receiver, MPI_Comm comm);
double* merge_array(int first_array_size, const double* first_array, int second_array_size, const double* second_array);
void merge_sort(int array_size, double* array);
void swap(double* a, double* b);
int is_sorted(int array_size, double* array, MPI_Comm comm);
int compare(const void* a, const void* b);

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
	MPI_OddEven_sort(NUMBER_OF_ELEMENTS, array, 0, MPI_COMM_WORLD);
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

int MPI_OddEven_sort(const int array_size, double* first_array, const int root, const MPI_Comm comm)
{
	double communication_time = 0.0, computation_time = 0.0;
	int rank, size, flag, all_flags;
	MPI_Comm_size(comm, &size);
	MPI_Comm_rank(comm, &rank);

	double* second_array = calloc(array_size / size, sizeof(double));

	double time = MPI_Wtime();
	MPI_Scatter(&first_array[0], array_size / size, MPI_DOUBLE, second_array, array_size / size, MPI_DOUBLE, root, comm);
	time = MPI_Wtime() - time;
	communication_time += time;

	time = MPI_Wtime();
	qsort(second_array, array_size / size, sizeof(double), compare);
	time = MPI_Wtime() - time;
	computation_time += time;

	time = MPI_Wtime();
	for (int index = 0; index < size; index++)
	{
		flag = is_sorted(array_size / size, second_array, comm);

		MPI_Allreduce(&flag, &all_flags, 1, MPI_INT, MPI_MIN, comm);

		if (all_flags == 1)
		{
			break;
		}

		if ((index + rank) % 2 == 0)
		{
			if (rank < size - 1)
			{
				exchange(array_size / size, second_array, rank, rank + 1, comm);
			}
		}
		else
		{
			if (rank > 0)
			{
				exchange(array_size / size, second_array, rank - 1, rank, comm);
			}
		}
		MPI_Barrier(comm);
	}
	time = MPI_Wtime() - time;
	computation_time += time;

	time = MPI_Wtime();
	MPI_Gather(second_array, array_size / size, MPI_DOUBLE, first_array, array_size / size, MPI_DOUBLE, root, comm);
	time = MPI_Wtime() - time;
	communication_time += time;

	printf("Processor number %d:\n-computation time: %lf\n-communication = %lf\n-total execution = %lf \n", rank, computation_time, communication_time, computation_time + communication_time);

	return MPI_SUCCESS;
}

void exchange(const int array_size, double* first_array, const int sender, const int receiver, const MPI_Comm comm)
{
	int rank;
	MPI_Comm_rank(comm, &rank);
	MPI_Status status;

	double* second_array = calloc(array_size, sizeof(double));
	double* merged_array;

	if (rank == sender)
	{
		MPI_Send(first_array, array_size, MPI_DOUBLE, receiver, 1, comm);
		MPI_Recv(second_array, array_size, MPI_DOUBLE, receiver, 2, comm, &status);

		merged_array = merge_array(array_size, first_array, array_size, second_array);

		for (int i = 0; i < array_size; i++)
		{
			first_array[i] = merged_array[i];
		}
	}
	else
	{
		MPI_Recv(second_array, array_size, MPI_DOUBLE, sender, 1, comm, &status);
		MPI_Send(first_array, array_size, MPI_DOUBLE, sender, 2, comm);
		merged_array = merge_array(array_size, first_array, array_size, second_array);
		for (int index = 0; index < array_size; index++)
		{
			first_array[index] = merged_array[index + array_size];
		}
	}
}

int is_sorted(const int array_size, double* array, const MPI_Comm comm)
{
	int rank, size, flag = 1;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);
	if (rank < size - 1)
	{
		MPI_Send(&array[array_size - 1], 1, MPI_DOUBLE, rank + 1, rank, comm);
	}
	if (rank > 0)
	{
		double temp;
		MPI_Status status;
		MPI_Recv(&temp, 1, MPI_DOUBLE, rank - 1, rank - 1, comm, &status);
		if (temp > array[0])
		{
			flag = 0;
		}
	}
	return flag;
}

void swap(double* a, double* b)
{
	const double temp = *a;
	*a = *b;
	*b = temp;
}

int compare(const void* a, const void* b)
{
	if (*(double*)a > *(double*)b)
	{
		return 1;
	}
	if (*(double*)a < *(double*)b)
	{
		return -1;
	}
	return 0;
}
