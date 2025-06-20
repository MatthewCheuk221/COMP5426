#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <sys/utsname.h>
#include <sys/resource.h>

#define SEED 8964

// validate command-line arguments
int validate_args(int argc, char *argv[], int *M, int *N, int *is_float) {
    if (argc != 4) {
        fprintf(stderr, "Error: Incorrect number of arguments.\n");
        fprintf(stderr, "Usage: %s M N datatype\n", argv[0]);
        return 0;
    }

    // convert M and N to integers
    char *end_pointer;

    *M = strtol(argv[1], &end_pointer, 10);
    if (*end_pointer != '\0' || *M <= 0) {
        fprintf(stderr, "Error: Invalid value for M. M, the number of columns, must be a positive integer.\n");
        return 0;
    }

    *N = strtol(argv[2], &end_pointer, 10);
    if (*end_pointer != '\0' || *N <= 0) {
        fprintf(stderr, "Error: Invalid value for N. N, the number of rows, must be a positive integer.\n");
        return 0;
    }

    // check datatype
    if (strcmp(argv[3], "float") == 0) {
        *is_float = 1;
    } else if (strcmp(argv[3], "double") == 0) {
        *is_float = 0;
    } else {
        fprintf(stderr, "Error: Invalid datatype. Datatype must be either 'float' or 'double'.\n");
        return 0;
    }

    return 1;
}

// initialize matrix with random values
void initialize_matrix(void *matrix, int N, int M, int is_float, int seed) {

    srand(seed);

    if (is_float) {
        float *matrix_float = (float*)matrix;
        for (int i = 0; i < N * M; i++) {
            matrix_float[i] = (float)rand() / RAND_MAX;
        }
    } else {
        double *matrix_double = (double*)matrix;
        for (int i = 0; i < N * M; i++) {
            matrix_double[i] = (double)rand() / RAND_MAX;
        }
    }
}

// sequential dot product computation
double sequential_dot_product(void *matrix, void *result, int N, int M, int is_float) {

    double start_time;

    // indices for matrix operations
    int i, j, k;

    if (is_float) {
        float *matrix_float = (float*)matrix;
        float *result_float = (float*)result;

        // initialize resulted matrix to zeros
        for (i = 0; i < M * M; i++) {
            result_float[i] = 0.0f;
        }

        start_time = MPI_Wtime();

        // compute dot products for all pairs (i,j) where i <= j by 1-layer loop unrolling with the unrolling factor of 4
        for (k = 0; k < N - 3; k += 4) {
            for (i = 0; i < M; i++) {
                for (j = i; j < M; j++) {
                    result_float[i * M + j] += matrix_float[k * M + i] * matrix_float[k * M + j]
                        + matrix_float[(k + 1) * M + i] * matrix_float[(k + 1) * M + j]
                        + matrix_float[(k + 2) * M + i] * matrix_float[(k + 2) * M + j]
                        + matrix_float[(k + 3) * M + i] * matrix_float[(k + 3) * M + j];
                }
            }
        }

        // handle remaining elements
        for (; k < N; k++) {
            for (i = 0; i < M; i++) {
                for (j = i; j < M; j++) {
                    result_float[i * M + j] += matrix_float[k * M + i] * matrix_float[k * M + j];
                }
            }
        }
    } else {
        double *matrix_double = (double*)matrix;
        double *result_double = (double*)result;

        // initialize resulted matrix to zeros
        for (i = 0; i < M * M; i++) {
            result_double[i] = 0.0;
        }

        start_time = MPI_Wtime();

        // compute dot products for all pairs (i,j) where i <= j by 1-layer loop unrolling with the unrolling factor of 4
        for (k = 0; k < N - 3; k += 4) {
            for (i = 0; i < M; i++) {
                for (j = i; j < M; j++) {
                    result_double[i * M + j] += matrix_double[k * M + i] * matrix_double[k * M + j]
                        + matrix_double[(k + 1) * M + i] * matrix_double[(k + 1) * M + j]
                        + matrix_double[(k + 2) * M + i] * matrix_double[(k + 2) * M + j]
                        + matrix_double[(k + 3) * M + i] * matrix_double[(k + 3) * M + j];
                }
            }
        }

        // handle remaining elements
        for (; k < N; k++) {
            for (i = 0; i < M; i++) {
                for (j = i; j < M; j++) {
                    result_double[i * M + j] += matrix_double[k * M + i] * matrix_double[k * M + j];
                }
            }
        }
    }

    return MPI_Wtime() - start_time;
}

// memory-distributing dot product computation
double distributed_dot_product(void *matrix, void *result, int N, int M, int is_float, int rank, int size) {

    double start_time, finish_time;

    // indices for matrix operations
    int i, j, k;

    // rows per process
    int rows_per_process = N / size;
    int rows_remainder = N % size;
    int rows = rows_per_process + (rank < rows_remainder ? 1 : 0);

    // memory for partial results and rows per process
    void *local_result = NULL;
    void *local_rows = NULL;

    // distribute rows from process 0 to all processes
    if (is_float) {
        float *matrix_float = (float*)matrix;
        local_rows = malloc(rows * M * sizeof(float));
        float *local_rows_float = (float*)local_rows;
        local_result = malloc(M * M * sizeof(float));
        float *local_result_float = (float*)local_result;

        // initialize local resulted matrix to zeros
        for (i = 0; i < M * M; i++) {
            local_result_float[i] = 0.0f;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        if (rank == 0) {
            printf("Starting workload allocation for memory-distributing execution...\n");

            // process 0 keeps its own rows
            memcpy(local_rows_float, matrix_float, rows * M * sizeof(float));
            printf("Process %d has kept %d rows for its own.\n", rank, rows);

            // process 0 sends the rows to other processes
            for (int proc = 1; proc < size; proc++) {
                int process_count_rows = rows_per_process + (proc < rows_remainder ? 1 : 0);
                int process_starting_row = proc * rows_per_process + (proc < rows_remainder ? proc : rows_remainder);
                MPI_Send(&matrix_float[process_starting_row * M], process_count_rows * M, MPI_FLOAT, proc, 0, MPI_COMM_WORLD);
            }
        } else {

            // other processes receive rows from process 0
            MPI_Recv(local_rows_float, rows * M, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d has received %d rows from Process 0.\n", rank, rows);
        }

        // compute partial dot products for all pairs (i,j) where i <= j by 1-layer loop unrolling with the unrolling factor of 4
        for (k = 0; k < rows - 3; k += 4) {
            for (i = 0; i < M; i++) {
                for (j = i; j < M; j++) {
                    local_result_float[i * M + j] += local_rows_float[k * M + i] * local_rows_float[k * M + j]
                        + local_rows_float[(k + 1) * M + i] * local_rows_float[(k + 1) * M + j]
                        + local_rows_float[(k + 2) * M + i] * local_rows_float[(k + 2) * M + j]
                        + local_rows_float[(k + 3) * M + i] * local_rows_float[(k + 3) * M + j];
                }
            }
        }

        // handle remaining elements
        for (; k < rows; k++) {
            for (i = 0; i < M; i++) {
                for (j = i; j < M; j++) {
                    local_result_float[i * M + j] += local_rows_float[k * M + i] * local_rows_float[k * M + j];
                }
            }
        }

        // reduce partial results to process 0
        float *result_float = (float*)result;
        MPI_Reduce(local_result_float, result_float, M * M, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
        double *matrix_double = (double*)matrix;
        local_rows = malloc(rows * M * sizeof(double));
        double *local_rows_double = (double*)local_rows;
        local_result = malloc(M * M * sizeof(double));
        double *local_result_double = (double*)local_result;

        // initialize local resulted matrix to zeros
        for (i = 0; i < M * M; i++) {
            local_result_double[i] = 0.0;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        if (rank == 0) {
            printf("Starting workload allocation for memory-distributing execution...\n");

            // process 0 keeps its own rows
            memcpy(local_rows_double, matrix_double, rows * M * sizeof(double));
            printf("Process %d has kept %d rows for its own.\n", rank, rows);

            // process 0 sends the rows to other processes
            for (int proc = 1; proc < size; proc++) {
                int process_count_rows = rows_per_process + (proc < rows_remainder ? 1 : 0);
                int process_starting_row = proc * rows_per_process + (proc < rows_remainder ? proc : rows_remainder);
                MPI_Send(&matrix_double[process_starting_row * M], process_count_rows * M, MPI_DOUBLE, proc, 0, MPI_COMM_WORLD);
            }
        } else {

            // other processes receive rows from process 0
            MPI_Recv(local_rows_double, rows * M, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Process %d has received %d rows from Process 0.\n", rank, rows);
        }

        // compute partial dot products for all pairs (i,j) where i <= j by 1-layer loop unrolling with the unrolling factor of 4
        for (k = 0; k < rows - 3; k += 4) {
            for (i = 0; i < M; i++) {
                for (j = i; j < M; j++) {
                    local_result_double[i * M + j] += local_rows_double[k * M + i] * local_rows_double[k * M + j]
                        + local_rows_double[(k + 1) * M + i] * local_rows_double[(k + 1) * M + j]
                        + local_rows_double[(k + 2) * M + i] * local_rows_double[(k + 2) * M + j]
                        + local_rows_double[(k + 3) * M + i] * local_rows_double[(k + 3) * M + j];
                }
            }
        }

        // handle remaining elements
        for (; k < rows; k++) {
            for (i = 0; i < M; i++) {
                for (j = i; j < M; j++) {
                    local_result_double[i * M + j] += local_rows_double[k * M + i] * local_rows_double[k * M + j];
                }
            }
        }

        // reduce partial results to process 0
        double *result_double = (double*)result;
        MPI_Reduce(local_result_double, result_double, M * M, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    finish_time = MPI_Wtime();

    free(local_rows);
    free(local_result);

    return finish_time - start_time;
}

int compare_results(void *seq_result, void *dis_result, int M, int is_float) {

    if (is_float) {
        float *seq_result_float = (float*)seq_result;
        float *dis_result_float = (float*)dis_result;

        for (int i = 0; i < M; i++) {
            for (int j = i; j < M; j++) {

                // handle error value close to zero
                float error = fabs(dis_result_float[i * M + j] - seq_result_float[i * M + j]) / fmaxf(fabs(seq_result_float[i * M + j]), 1.0f);

                // tolerance for float calculations (1e-5 or 0.001%)
                if (error > 1e-5f) {
                    printf("Mismatch at (%d, %d): Sequential: %.7f      Memory-distributing: %.7f\n", i, j, seq_result_float[i * M + j], dis_result_float[i * M + j]);
                    printf("Error: %e\n\n", error);
                    return 0;
                }
            }
        }
    } else {
        double *seq_result_double = (double*)seq_result;
        double *dis_result_double = (double*)dis_result;

        for (int i = 0; i < M; i++) {
            for (int j = i; j < M; j++) {

                // handle error value close to zero
                float error = fabs(dis_result_double[i * M + j] - seq_result_double[i * M + j]) / fmaxf(fabs(seq_result_double[i * M + j]), 1.0f);

                // tolerance for double calculations (1e-10 or 0.000000001%)
                if (error > 1e-10) {
                    printf("Mismatch at (%d, %d): Sequential: %.15lf      Memory-distributing: %.15lf\n", i, j, seq_result_double[i * M + j], dis_result_double[i * M + j]);
                    printf("Error: %e\n\n", error);
                    return 0;
                }
            }
        }
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int M, N;
    int is_float;
    double seq_time, dis_time;
    void *matrix = NULL;
    void *seq_result = NULL;
    void *dis_result = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (!validate_args(argc, argv, &M, &N, &is_float)) {
        MPI_Finalize();
        return 1;
    }

    // process 0 allocates and initializes matrices
    if (rank == 0) {
        int matrix_size = N * M;
        int result_size = M * M;

        if (is_float) {
            matrix = malloc(matrix_size * sizeof(float));
            seq_result = malloc(result_size * sizeof(float));
            dis_result = calloc(result_size, sizeof(float));
        } else {
            matrix = malloc(matrix_size * sizeof(double));
            seq_result = malloc(result_size * sizeof(double));
            dis_result = calloc(result_size, sizeof(double));
        }

        if (!matrix || !seq_result || !dis_result) {
            fprintf(stderr, "Error: Memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        initialize_matrix(matrix, N, M, is_float, SEED);
    }

    // single process mode
    if (size == 1) {
        seq_time = sequential_dot_product(matrix, seq_result, N, M, is_float);
        printf("Sequential execution time: %f seconds\n\n", seq_time);
    } else {

        // multi-process mode
        // sequential computation by process 0
        if (rank == 0) {
            seq_time = sequential_dot_product(matrix, seq_result, N, M, is_float);
            printf("Sequential execution time: %f second(s)\n\n", seq_time);
        }

        // memory-distributing computation by all processes
        dis_time = distributed_dot_product(matrix, dis_result, N, M, is_float, rank, size);

        if (rank == 0) {

            // calculate speedup
            printf("\nMemory-distributing execution time: %f second(s)\n\n", dis_time);
            printf("Speedup: %.3f\n\n", seq_time / dis_time);

            // compare results
            if (compare_results(seq_result, dis_result, M, is_float)) {
                printf("Sequential and memory-distributing results match.\n\n");
            } else {
                printf("Sequential and memory-distributing results do not match.\n\n");
            }
        }
    }

    // de-allocation
    if (rank == 0) {
        free(matrix);
        free(seq_result);
        free(dis_result);

        // get max RAM usage
        struct rusage max_ram_usage;
        getrusage(RUSAGE_SELF, &max_ram_usage);

        // determine the OS for the measuring unit of max RAM usage
        struct utsname system_info;
        if (uname(&system_info) == 0) {
            if (strstr(system_info.sysname, "Darwin")) {
                printf("Maximum RAM usage: %ld bytes\n\n", max_ram_usage.ru_maxrss);
            } else if (strstr(system_info.sysname, "Linux")) {
                printf("Maximum RAM usage: %ld KB\n\n", max_ram_usage.ru_maxrss);
            } else {
                printf("Unknown OS: %s\n", system_info.sysname);
                printf("Maximum RAM usage: %ld KB\n\n", max_ram_usage.ru_maxrss);
            }
        } else {
            printf("Unable to determine the OS.\n");
            printf("Maximum RAM usage: %ld KB\n\n", max_ram_usage.ru_maxrss);
        }
    }

    MPI_Finalize();
    return 0;
}
