#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include "mpi.h"

const int NUMBER_OF_ELEMENTS = 1048576;

void compare_low(int bit, int* array, int array_size, int rank, double* comp_time, double* comm_time);
void compare_high(int bit, int* array, int array_size, int rank, double* comp_time, double* comm_time);
int comparator(const void* a, const void* b);
void bitonic_sort(int rank, int processors_number, int* array, int array_size);

int main(int argc, char* argv[])
{
	int process_rank = 0;
	int processors_number = 0;
	int* array = NULL;
	int array_size = 0;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &processors_number);
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
	array_size = NUMBER_OF_ELEMENTS / processors_number;
	array = (int*)malloc(array_size * sizeof(int));

	srand(time(NULL));
	for (int index = 0; index < array_size; index++)
	{
		array[index] = rand() % NUMBER_OF_ELEMENTS;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (process_rank == 0)
	{
		printf("Processors: %d\n", processors_number);
	}

	bitonic_sort(process_rank, processors_number, array, array_size);

	MPI_Finalize();
	return 0;
}

void bitonic_sort(int rank, int processors_number, int* array, int array_size)
{
	double timer_start = MPI_Wtime();
	double computation_time = 0, communication_time = 0;
	double time = MPI_Wtime();
	qsort(array, array_size, sizeof(int), comparator);
	time = MPI_Wtime() - time;
	computation_time += time;

	const int dimensions = (int)(log2(processors_number));
	for (int index = 0; index < dimensions; index++)
	{
		for (int j = index; j >= 0; j--)
		{
			double aux_comp_time = 0, aux_comm_time = 0;
			if (((rank >> (index + 1)) % 2 == 0 && (rank >> j) % 2 == 0) || ((rank >> (index + 1)) % 2 != 0 && (rank >>j) % 2 != 0))
			{
				compare_low(j, array, array_size, rank, &aux_comp_time, &aux_comm_time);
				communication_time += aux_comm_time;
				communication_time += aux_comp_time;
			}
			else
			{
				compare_high(j, array, array_size, rank, &aux_comp_time, &aux_comm_time);
				communication_time += aux_comm_time;
				communication_time += aux_comp_time;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	printf("Processor number %d:\n-computation time: %lf\n-communication = %lf\n-total execution = %lf \n", rank, computation_time, communication_time, computation_time + communication_time);
	free(array);
}

void compare_low(const int bit, int* array, const int array_size, int rank, double* comp_time, double* comm_time)
{
	int min;
	int send_counter = 0;
	int* buffer_send = malloc((array_size + 1) * sizeof(int));
	double time = MPI_Wtime();
	MPI_Send(&array[array_size - 1], 1,MPI_INT, rank ^ (1 << bit), 0, MPI_COMM_WORLD);
	int* buffer_receiver = malloc((array_size + 1) * sizeof(int));
	MPI_Recv(&min, 1,MPI_INT, rank ^ (1 << bit), 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	time = MPI_Wtime() - time;
	*comm_time += time;

	time = MPI_Wtime();
	for (int index = 0; index < array_size; index++)
	{
		if (array[index] > min)
		{
			buffer_send[send_counter + 1] = array[index];
			send_counter++;
		}
		else
		{
			break; //saves a lot of cycles
		}
	}
	time = MPI_Wtime() - time;
	*comp_time += time;

	buffer_send[0] = send_counter;
	time = MPI_Wtime();
	MPI_Send(buffer_send, send_counter,MPI_INT, rank ^ (1 << bit), 0, MPI_COMM_WORLD);
	MPI_Recv(buffer_receiver, array_size,MPI_INT, rank ^ (1 << bit), 0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	time = MPI_Wtime() - time;
	*comm_time += time;

	time = MPI_Wtime();
	for (int index = 1; index < buffer_receiver[0] + 1; index++)
	{
		if (array[array_size - 1] < buffer_receiver[index])
		{
			array[array_size - 1] = buffer_receiver[index];
		}
		else
		{
			break; //saves a lot of cycles
		}
	}
	qsort(array, array_size, sizeof(int), comparator);
	time = MPI_Wtime() - time;
	*comp_time += time;
	free(buffer_send);
	free(buffer_receiver);
}


void compare_high(const int bit, int* array, const int array_size, const int rank, double* comp_time, double* comm_time)
{
	int max;
	int* buffer_receiver = malloc((array_size + 1) * sizeof(int));
	double time = MPI_Wtime();
	MPI_Recv(&max, 1,MPI_INT, rank ^ (1 << bit), 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	int send_counter = 0;
	int* buffer_send = malloc((array_size + 1) * sizeof(int));
	MPI_Send(&array[0], 1,MPI_INT, rank ^ (1 << bit), 0,MPI_COMM_WORLD);
	time = MPI_Wtime() - time;
	*comm_time += time;

	time = MPI_Wtime();
	for (int index = 0; index < array_size; index++)
	{
		if (array[index] < max)
		{
			buffer_send[send_counter + 1] = array[index];
			send_counter++;
		}
		else
		{
			break; //saves a lot of cycles
		}
	}
	time = MPI_Wtime() - time;
	*comp_time += time;

	time = MPI_Wtime();
	MPI_Recv(buffer_receiver, array_size,MPI_INT, rank ^ (1 << bit), 0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	const int receive_counter = buffer_receiver[0];
	buffer_send[0] = send_counter;
	MPI_Send(buffer_send, send_counter,MPI_INT, rank ^ (1 << bit), 0, MPI_COMM_WORLD);
	time = MPI_Wtime() - time;
	*comm_time += time;

	time = MPI_Wtime();
	for (int index = 1; index < receive_counter + 1; index++)
	{
		if (buffer_receiver[index] > array[0])
		{
			array[0] = buffer_receiver[index];
		}
		else
		{
			break; //saves a lot of cycles
		}
	}
	qsort(array, array_size, sizeof(int), comparator);
	time = MPI_Wtime() - time;
	*comp_time += time;
	free(buffer_send);
	free(buffer_receiver);
}

int comparator(const void* a, const void* b)
{
	return *(int*)a - *(int*)b;
}
