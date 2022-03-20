# Parallel sorting algorithms written in plain C using Microsoft MPI

#### This repo includes the following parallel algorithms:
- Bitonic sort: https://en.wikipedia.org/wiki/Bitonic_sorter
- Bucket sort: https://en.wikipedia.org/wiki/Bucket_sort
- Direct(selection) sort: https://en.wikipedia.org/wiki/Selection_sort
- Odd-Even sort: https://en.wikipedia.org/wiki/Odd%E2%80%93even_sort
- Shell sort: https://en.wikipedia.org/wiki/Shellsort

#### How to run:
- compile the code (if you are using Visual Studio, you can find the binary under Debug folder)
- ```run mpiexec -n x Name``` where n is the number of physical cores and Name is the name of the binary.
- E.g.: ```C:\Users\Ionut\source\repos\DirectSort\Debug>mpiexec -n 4 DirectSort```

The detailed comparison is attached in ```comparison.pdf```
