Disclaimer: These materials are what you will be expected to deliver in this Unit of Study and shall be regarded as indicative only.

To compile assignment_1.c

gcc assignment_1.c -o assignment_1 -pthread -D_GNU_SOURCE -Wall -pedantic

To run assignment_1.c

./assignment_1 <matrices_A_and_D_rows> <matrix_A_columns_and_matrix_B_rows> <matrix_B_columns_and_matrix_C_rows> <matrices_C_and_D_columns> <float/double> <number_of_threads>

To compile assignment_2.c

mpicc assignment_2.c -o assignment_2 -lm -Wall -pedantic

To run assignment_2.c

mpirun -np <number_of_processes> (--oversubscribe if the number of processes to be created is more than the number of processors) ./assignment_2 <number_of_rows> <number_of_columns> <float/double>

Number of processors available for pthreads in soitpa10008/soitpa10009/soitpa10010: 4

Number of slots available for MPI in soitpa10008/soitpa10009/soitpa10010: 2
