#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const int ARRAY_SIZE = 1000000;

void create_random_array(int** array, const int n)
{
	int* arr = malloc(n * sizeof(int));

	for (int i = 0; i < n; i++)
	{
		arr[i] = rand() % n;
	}
	*array = arr;
}

void merge_sort(const int current_core, const int local_array_size, int* local_array, const int* tmp, int* tmp2, const int compare_to)
{
	int i, j, k;
	if (current_core < compare_to && tmp[0] < local_array[local_array_size - 1])
	{
		j = k = i = 0;
		while (k < local_array_size)
		{
			if (tmp[j] < local_array[i])
			{
				tmp2[k++] = tmp[j++];
			}
			else
			{
				tmp2[k++] = local_array[i++];
			}
		}
		memcpy(local_array, tmp2, local_array_size * sizeof(int));
	}
	else if (current_core > compare_to && tmp[local_array_size - 1] > local_array[0])
	{
		j = k = i = local_array_size - 1;
		while (k >= 0)
		{
			if (tmp[j] > local_array[i])
			{
				tmp2[k--] = tmp[j--];
			}
			else
			{
				tmp2[k--] = local_array[i--];
			}
		}
		memcpy(local_array, tmp2, local_array_size * sizeof(int));
	}
}

void odd_even_compare_split(const int current_core, const int core_numbers, const int local_array_size, int* local_array, int* tmp, int* tmp2)
{
	int flag = core_numbers;
	while (--flag)
	{
		int compareTo;
		if (flag % 2)
		{
			compareTo = (current_core % 2 == 0) ? current_core + 1 : current_core - 1;
		}
		else
		{
			compareTo = (current_core % 2 == 0) ? current_core - 1 : current_core + 1;
			if (compareTo == -1 || compareTo == core_numbers)
			{
				compareTo = MPI_PROC_NULL;
			}
		}

		MPI_Sendrecv(local_array, local_array_size, MPI_INT, compareTo, 10 + flag, tmp, local_array_size, MPI_INT, compareTo, 10 + flag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		merge_sort(current_core, local_array_size, local_array, tmp, tmp2, compareTo);
	}
}

void divide_and_move_closer(const int current_core, const int core_numbers, const int array_size, int* local_array, int* tmp, int* tmp2)
{
	int divider = core_numbers;
	int compare_to = divider - current_core - 1;
	divider /= 2;
	while (divider > 0)
	{
		MPI_Sendrecv(local_array, array_size, MPI_INT, compare_to, divider, tmp, array_size, MPI_INT, compare_to, divider, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		merge_sort(current_core, array_size, local_array, tmp, tmp2, compare_to);
		const int n = current_core / divider;
		compare_to = (n + 1) * divider - current_core % divider - 1;
		divider /= 2;
	}
}

void do_shell_sort_on_own_array(const int items_per_core, int* local_array)
{
	for (int divider = items_per_core / 2; divider > 0; divider /= 2)
	{
		for (int i = divider; i < items_per_core; i++)
		{
			const int temp = local_array[i];
			int j;
			for (j = i; j >= divider && temp < local_array[j - divider]; j -= divider)
			{
				local_array[j] = local_array[j - divider];
			}
			local_array[j] = temp;
		}
	}
}


void shell_sort(int current_core, int core_numbers)
{
	int* array = NULL, * local_array = NULL, * tmp = NULL, * tmp2 = NULL;
	const int items_per_core = ARRAY_SIZE / core_numbers;

	if (current_core == 0)
	{
		create_random_array(&array, items_per_core * core_numbers);
	}
	
	if (core_numbers > 1)
	{
		local_array = malloc(items_per_core * sizeof(int));
		tmp = malloc(items_per_core * sizeof(int));
		tmp2 = malloc(items_per_core * sizeof(int));
	}

	double start_time = 0, communication_time = 0, computation_time = 0, aux_time = 0;
	if (current_core == 0)
	{
		start_time = MPI_Wtime();
	}

	if (core_numbers > 1)
	{
		aux_time = MPI_Wtime();
		MPI_Scatter(array, items_per_core, MPI_INT, local_array, items_per_core, MPI_INT, 0, MPI_COMM_WORLD);
		aux_time = MPI_Wtime() - aux_time;
		communication_time += aux_time;
	}
	else
	{
		local_array = array;
	}

	aux_time = MPI_Wtime();
	do_shell_sort_on_own_array(items_per_core, local_array);
	divide_and_move_closer(current_core, core_numbers, items_per_core, local_array, tmp, tmp2);
	odd_even_compare_split(current_core, core_numbers, items_per_core, local_array, tmp, tmp2);
	aux_time = MPI_Wtime() - aux_time;
	computation_time += aux_time;
	
	if (core_numbers > 1)
	{
		aux_time = MPI_Wtime();
		MPI_Gather(local_array, items_per_core, MPI_INT, array, items_per_core, MPI_INT, 0, MPI_COMM_WORLD);
		aux_time = MPI_Wtime() - aux_time;
		communication_time += aux_time;
	}
	
	if (current_core == 0)
	{
		free(array);
	}

	if (core_numbers > 1)
	{
		free(local_array);
		free(tmp);
		free(tmp2);
	}
	printf("Processor number %d:\n-computation time: %lf\n-communication = %lf\n-total execution = %lf \n", current_core, computation_time, communication_time, computation_time + communication_time);

}

int main(int argc, char* argv[])
{
	int current_core, core_numbers;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &current_core);
	MPI_Comm_size(MPI_COMM_WORLD, &core_numbers);

	int check = core_numbers;
	while (check > 1)
	{
		if (check % 2 != 0)
		{
			fprintf(stderr, "Numbers of cores must power of 2");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		check /= 2;
	}

	const double start_time = MPI_Wtime();
	shell_sort(current_core, core_numbers);
	printf("All processing took %lf\n", MPI_Wtime() - start_time);

	MPI_Finalize();
	return 0;
}
