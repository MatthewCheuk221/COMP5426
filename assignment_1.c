// #define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <sys/utsname.h>
#include <sys/resource.h>

#define SEED 8964
#define BLOCK_SIZE 64

pthread_mutex_t core_tracking_mutex = PTHREAD_MUTEX_INITIALIZER;
int* threads_per_core = NULL;  // array to track threads per core

typedef struct {
    void* data;
    int rows;
    int cols;
} matrix_t;

typedef struct {
    int** s;      // optimal split positions
    int* dims;    // dimensions of matrices
    int num_matrices;    // number of matrices
} matrix_chain_t;

// synchronization structure
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int total;
    int phase;
} sync_point_t;

typedef struct {
    matrix_t* matrices;
    matrix_t* result;
    matrix_t* temp;        // array of temporary matrices
    int start_row;         // start row for this thread
    int end_row;           // end row for this thread
    int is_float;
    int is_par;
    matrix_chain_t* chain;
    sync_point_t* sync;    // synchronization
    int num_threads;
    int thread_id;
    int block_size;
} thread_data_t;


void* matrix_multiply_thread(void* arg);
void sequential_matrix_multiplication(matrix_t* matrices, matrix_t* result, matrix_t* temp, int is_float, matrix_chain_t* chain);
matrix_chain_t* matrix_chain_order(int* dims, int n);
void multiply_matrices(void* A, void* B, void* C, int m, int k, int n, int is_float, int is_par, int start_row, int end_row, int block_size);
void initialize_matrix(matrix_t* matrix, int is_float);
int compare_matrices(void* num_1, void* num_2, int rows, int cols, int is_float);
void free_matrix_chain(matrix_chain_t* chain);
void print_system_info(void);

void sync_point_init(sync_point_t* sync, int total) {
    pthread_mutex_init(&sync->mutex, NULL);
    pthread_cond_init(&sync->cond, NULL);
    sync->count = 0;
    sync->total = total;
    sync->phase = 0;
}

void sync_point_wait(sync_point_t* sync) {
    pthread_mutex_lock(&sync->mutex);
    int arrival_phase = sync->phase;
    sync->count++;

    if (sync->count == sync->total) {
        sync->count = 0;
        sync->phase = arrival_phase + 1;
        pthread_cond_broadcast(&sync->cond);
    } else {
        while (arrival_phase == sync->phase) {
            pthread_cond_wait(&sync->cond, &sync->mutex);
        }
    }
    pthread_mutex_unlock(&sync->mutex);
}

void sync_point_destroy(sync_point_t* sync) {
    pthread_mutex_destroy(&sync->mutex);
    pthread_cond_destroy(&sync->cond);
}

void print_system_info(void) {
    struct utsname buffer;

    if (uname(&buffer) != 0) {
        perror("uname");
        return;
    }

    printf("\nSystem Information:\n");
    printf("  System: %s\n", buffer.sysname);
    printf("  Node: %s\n", buffer.nodename);
    printf("  Release: %s\n", buffer.release);
    printf("  Version: %s\n", buffer.version);
    printf("  Machine: %s\n", buffer.machine);
    printf("  Number of available processors: %ld\n", sysconf(_SC_NPROCESSORS_ONLN));

    // run lscpu and capture the output
    printf("\nCPU Information (from lscpu):\n");
    FILE* fp = popen("lscpu", "r");
    if (fp != NULL) {
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), fp) != NULL) {
            printf("  %s", buffer);
        }
        pclose(fp);
    } else {
        printf("Warning: Unable to run lscpu.\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int m, k, l, n, num_threads;
    char* datatype;
    int is_float; // 1 for float, 0 for double
    matrix_t matrices[3]; // A, B, C
    matrix_t result_seq, result_par;
    matrix_t* temp_matrix; // for temporary result T

    print_system_info();

    if (argc != 7) {
        fprintf(stderr, "Error: Incorrect number of arguments.\n");
        fprintf(stderr, "Usage: ./a.out m k l n datatype number_of_threads\n");
        return 1;
    }

    m = atoi(argv[1]);
    k = atoi(argv[2]);
    l = atoi(argv[3]);
    n = atoi(argv[4]);
    datatype = argv[5];
    num_threads = atoi(argv[6]);

    if (m <= 0 || k <= 0 || l <= 0 || n <= 0) {
        fprintf(stderr, "Error: Matrix dimensions must be positive integers.\n");
        return 1;
    }

    if (strcmp(datatype, "float") == 0) {
        is_float = 1;
    } else if (strcmp(datatype, "double") == 0) {
        is_float = 0;
    } else {
        fprintf(stderr, "Error: Datatype must be either 'float' or 'double'.\n");
        return 1;
    }

    if (num_threads <= 0 || num_threads > 64) {
        fprintf(stderr, "Error: Number of threads must be between 1 and 64.\n");
        return 1;
    }

    if (m < num_threads || k < num_threads || l < num_threads || n < num_threads) {
        fprintf(stderr, "Error: Number of threads must be less than number of rows and columns of any matrix.\n");
        return 1;
    }

    threads_per_core = calloc(sysconf(_SC_NPROCESSORS_ONLN), sizeof(int));
    if (!threads_per_core) {
        fprintf(stderr, "Error: Memory allocation failed for thread tracking.\n");
        return 1;
    }

    if (num_threads > sysconf(_SC_NPROCESSORS_ONLN)) {
        printf("Warning: %d threads have been created but only %ld cores are available. Performance may not be scaled as expected.\n\n", num_threads, sysconf(_SC_NPROCESSORS_ONLN));
    }

    // matrix dimensions for the chain algorithm
    int dims[4] = {m, k, l, n};
    matrix_chain_t* chain = matrix_chain_order(dims, 4);
    if (chain->s[1][3] == 1) {
        printf("Optimal multiplication order: T = A × B and then D = T × C\n\n");
    } else {
        printf("Optimal multiplication order: T = B × C and then D = A × T\n\n");
    }

    // allocate memory for matrices
    matrices[0].rows = m; matrices[0].cols = k; // A
    matrices[1].rows = k; matrices[1].cols = l; // B
    matrices[2].rows = l; matrices[2].cols = n; // C
    result_seq.rows = m; result_seq.cols = n;   // D_seq
    result_par.rows = m; result_par.cols = n;   // D_par

    if (is_float) {
        matrices[0].data = malloc(m * k * sizeof(float)); // A
        matrices[1].data = malloc(k * l * sizeof(float)); // B
        matrices[2].data = malloc(l * n * sizeof(float)); // C
        result_seq.data = malloc(m * n * sizeof(float));  // D_seq
        result_par.data = malloc(m * n * sizeof(float));  // D_par
    } else {
        matrices[0].data = malloc(m * k * sizeof(double)); // A
        matrices[1].data = malloc(k * l * sizeof(double)); // B
        matrices[2].data = malloc(l * n * sizeof(double)); // C
        result_seq.data = malloc(m * n * sizeof(double));  // D_seq
        result_par.data = malloc(m * n * sizeof(double));  // D_par
    }

    // allocate temporary matrices for calculations (we need at most 2 for 3 matrices)
    temp_matrix = (matrix_t*)malloc(2 * sizeof(matrix_t));
    int temp_rows, temp_cols;
    if (chain->s[1][3] == 1) {
        // T = A × B: temp is A × B with dimension m×l
        temp_rows = m;
        temp_cols = l;
    } else {
        // T = B × C: temp is B × C with dimension k×n
        temp_rows = k;
        temp_cols = n;
    }

    temp_matrix[0].rows = temp_rows;
    temp_matrix[0].cols = temp_cols;

    // final result D with dimension m×n
    temp_matrix[1].rows = m;
    temp_matrix[1].cols = n;

    if (is_float) {
        temp_matrix[0].data = malloc(temp_rows * temp_cols * sizeof(float));
        temp_matrix[1].data = malloc(m * n * sizeof(float));
    } else {
        temp_matrix[0].data = malloc(temp_rows * temp_cols * sizeof(double));
        temp_matrix[1].data = malloc(m * n * sizeof(double));
    }

    // de-allocation check
    if (!matrices[0].data || !matrices[1].data || !matrices[2].data ||
        !result_seq.data || !result_par.data || !temp_matrix[0].data || !temp_matrix[1].data) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        for (int i = 0; i < 3; i++) {
            if (matrices[i].data) free(matrices[i].data);
        }
        if (result_seq.data)
            free(result_seq.data);
        if (result_par.data)
            free(result_par.data);
        if (temp_matrix) {
            for (int i = 0; i < 2; i++) {
                if (temp_matrix[i].data)
                    free(temp_matrix[i].data);
            }
            free(temp_matrix);
        }
        free_matrix_chain(chain);
        if (threads_per_core)
            free(threads_per_core);
        return 1;
    }

    srand(SEED);

    for (int i = 0; i < 3; i++) {
        initialize_matrix(&matrices[i], is_float);
    }

    // initialize result and temporary matrices to zero
    size_t result_size = result_par.rows * result_par.cols * (is_float ? sizeof(float) : sizeof(double));
    memset(result_seq.data, 0, result_size);
    memset(result_par.data, 0, result_size);

    for (int i = 0; i < 2; i++) {
        size_t temp_size = temp_matrix[i].rows * temp_matrix[i].cols * (is_float ? sizeof(float) : sizeof(double));
        memset(temp_matrix[i].data, 0, temp_size);
    }

    // start sequential matrix multiplication
    struct timespec start_seq, end_seq;
    clock_gettime(CLOCK_REALTIME, &start_seq);
    sequential_matrix_multiplication(matrices, &result_seq, temp_matrix, is_float, chain);
    clock_gettime(CLOCK_REALTIME, &end_seq);
    double seq_time = (end_seq.tv_sec - start_seq.tv_sec) + (end_seq.tv_nsec - start_seq.tv_nsec) / 1e10;
    printf("Sequential execution time: %.6f second(s)\n\n", seq_time);

    // start parallel matrix multiplication
    printf("Starting Thread-to-Core allocation for parallel execution...\n");
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t* thread_data = malloc(num_threads * sizeof(thread_data_t));
    sync_point_t sync_point;

    sync_point_init(&sync_point, num_threads);

    for (int i = 0; i < num_threads; i++) {
        int rows_per_thread = m / num_threads;
        int start_row = i * rows_per_thread;
        int end_row = (i == num_threads - 1) ? m : (i + 1) * rows_per_thread;

        thread_data[i].matrices = matrices;
        thread_data[i].result = &result_par;
        thread_data[i].temp = temp_matrix;
        thread_data[i].start_row = start_row;
        thread_data[i].end_row = end_row;
        thread_data[i].is_float = is_float;
        thread_data[i].is_par = 1;
        thread_data[i].chain = chain;
        thread_data[i].sync = &sync_point;
        thread_data[i].num_threads = num_threads;
        thread_data[i].thread_id = i;
        thread_data[i].block_size = BLOCK_SIZE;

        pthread_create(&threads[i], NULL, matrix_multiply_thread, &thread_data[i]);
    }

    struct timespec start_par, end_par;
    clock_gettime(CLOCK_REALTIME, &start_par);

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_REALTIME, &end_par);

    printf("\nThread-to-Core distribution:\n");
    for (int i = 0; i < sysconf(_SC_NPROCESSORS_ONLN); i++) {
        printf("Core %d: %d thread(s)\n", i, threads_per_core[i]);
    }

    double par_time = (end_par.tv_sec - start_par.tv_sec) + (end_par.tv_nsec - start_par.tv_nsec) / 1e10;
    printf("\nParallel execution time: %.6f second(s)\n\n", par_time);
    printf("Speedup: %.3f\n\n", seq_time / par_time);

    if (compare_matrices(result_seq.data, result_par.data, m, n, is_float)) {
        printf("Sequential and parallel results match.\n\n");
    } else {
        printf("Sequential and parallel results do not match.\n\n");
    }

    // get max RAM usage
    struct rusage max_usage;
    getrusage(RUSAGE_SELF, &max_usage);

    // determine the OS for the measuring unit of max RAM usage
    struct utsname sys_info;
    if (uname(&sys_info) == 0) {
        if (strstr(sys_info.sysname, "Darwin")) {
            printf("Maximum RAM usage: %ld bytes\n\n", max_usage.ru_maxrss);
        } else if (strstr(sys_info.sysname, "Linux")) {
            printf("Maximum RAM usage: %ld KB\n\n", max_usage.ru_maxrss);
        } else {
            printf("Unknown OS: %s\n", sys_info.sysname);
            printf("Maximum RAM usage: %ld KB\n\n", max_usage.ru_maxrss);
        }
    } else {
        printf("Warning: Unable to determine the OS.\n");
        printf("Maximum RAM usage: %ld KB\n\n", max_usage.ru_maxrss);
    }

    // de-allocation
    for (int i = 0; i < 3; i++) {
        free(matrices[i].data);
    }
    free(result_seq.data);
    free(result_par.data);

    for (int i = 0; i < 2; i++) {
        free(temp_matrix[i].data);
    }
    free(temp_matrix);
    free(threads);
    free(thread_data);
    free_matrix_chain(chain);
    sync_point_destroy(&sync_point);
    return 0;
}

// determine the order of matrix multiplication chain by dynamic programming
matrix_chain_t* matrix_chain_order(int* dims, int n) {
    matrix_chain_t* chain = malloc(sizeof(matrix_chain_t));
    chain->dims = malloc(n * sizeof(int));
    memcpy(chain->dims, dims, n * sizeof(int));
    chain->num_matrices = n - 1;

    // allocate memory for m and s tables
    // long long for m
    long long** m = malloc(n * sizeof(long long*));
    chain->s = malloc(n * sizeof(int*));

    for (int i = 0; i < n; i++) {
        m[i] = malloc(n * sizeof(long long));
        chain->s[i] = malloc(n * sizeof(int));
        for (int j = 0; j < n; j++) {
            m[i][j] = (i == j) ? 0 : LLONG_MAX;
            chain->s[i][j] = 0;
        }
    }

    // cost calculation
    for (int len = 2; len <= n - 1; len++) {
        for (int i = 1; i <= n - len; i++) {
            int j = i + len - 1;
            for (int k = i; k < j; k++) {
                long long cost = m[i][k] + m[k+1][j] + (long long)dims[i-1] * dims[k] * dims[j];
                if (cost < m[i][j]) {
                    m[i][j] = cost;
                    chain->s[i][j] = k;
                }
            }
        }
    }

    printf("Minimum times of scalar multiplication: %lld\n\n", m[1][n-1]);

    for (int i = 0; i < n; i++)
        free(m[i]);
    free(m);
    return chain;
}

void free_matrix_chain(matrix_chain_t* chain) {
    if (chain) {
        if (chain->dims) free(chain->dims);
        if (chain->s) {
            for (int i = 0; i < chain->num_matrices + 1; i++) {
                if (chain->s[i]) free(chain->s[i]);
            }
            free(chain->s);
        }
        free(chain);
    }
}

// thread function
void* matrix_multiply_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;

    // set CPU affinity for this thread
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    // pin thread to specific core base on thread_id
    // ensure each thread runs on a different core if available
    int core_id = data->thread_id % sysconf(_SC_NPROCESSORS_ONLN);
    CPU_SET(core_id, &cpuset);

    pthread_mutex_lock(&core_tracking_mutex);
    threads_per_core[core_id]++;
    pthread_mutex_unlock(&core_tracking_mutex);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
        fprintf(stderr, "Error: Thread %d failed to set affinity to core %d.\n", data->thread_id, core_id);
    } else {
        printf("Thread %d has been pinned to Core %d.\n", data->thread_id, core_id);
    }

    matrix_chain_t* chain = data->chain;

    if (chain->s[1][3] == 1) {
        // for the order T = A × B and then D = T × C:
        // T = A × B (m×k × k×l = m×l)
        multiply_matrices(
                data->matrices[0].data,
                data->matrices[1].data,
                data->temp[0].data,
                data->matrices[0].rows,
                data->matrices[0].cols,
                data->matrices[1].cols,
                data->is_float,
                data->is_par,
                data->start_row,
                data->end_row,
                data->block_size
        );

        // wait for all threads before second multiplication
        sync_point_wait(data->sync);

        // D = T × C (m×l × l×n = m×n)
        multiply_matrices(
                data->temp[0].data,
                data->matrices[2].data,
                data->result->data,
                data->temp[0].rows,
                data->temp[0].cols,
                data->matrices[2].cols,
                data->is_float,
                data->is_par,
                data->start_row,
                data->end_row,
                data->block_size
        );
    } else {
        // for the order T = B × C and then D = A × T:
        // T = B × C (k×l × l×n = k×n)
        int k = data->matrices[1].rows;
        int start_row_b = (data->start_row * k) / data->matrices[0].rows;
        int end_row_b = (data->end_row * k) / data->matrices[0].rows;
        if (end_row_b > k) end_row_b = k;

        multiply_matrices(
                data->matrices[1].data,
                data->matrices[2].data,
                data->temp[0].data,
                data->matrices[1].rows,
                data->matrices[1].cols,
                data->matrices[2].cols,
                data->is_float,
                data->is_par,
                start_row_b,
                end_row_b,
                data->block_size
        );

        // wait for all threads before second multiplication
        sync_point_wait(data->sync);

        // D = A × T (m×k × k×n = m×n)
        multiply_matrices(
                data->matrices[0].data,
                data->temp[0].data,
                data->result->data,
                data->matrices[0].rows,
                data->matrices[0].cols,
                data->temp[0].cols,
                data->is_float,
                data->is_par,
                data->start_row,
                data->end_row,
                data->block_size
        );
    }
    return NULL;
}

// sequential matrix multiplication
void sequential_matrix_multiplication(matrix_t* matrices, matrix_t* result, matrix_t* temp, int is_float, matrix_chain_t* chain) {

    if (chain->s[1][3] == 1) {
        // for the order T = A × B and then D = T × C:
        // T = A × B (m×k × k×l = m×l)
        multiply_matrices(
                matrices[0].data,
                matrices[1].data,
                temp[0].data,
                matrices[0].rows,
                matrices[0].cols,
                matrices[1].cols,
                is_float,
                0,
                0,
                matrices[0].rows,
                BLOCK_SIZE
        );

        // D = T × C (m×l × l×n = m×n)
        multiply_matrices(
                temp[0].data,
                matrices[2].data,
                result->data,
                temp[0].rows,
                temp[0].cols,
                matrices[2].cols,
                is_float,
                0,
                0,
                temp[0].rows,
                BLOCK_SIZE
        );
    } else {
        // for the order T = B × C and then D = A × T:
        // T = B × C (k×l × l×n = k×n)
        multiply_matrices(
                matrices[1].data,
                matrices[2].data,
                temp[0].data,
                matrices[1].rows,
                matrices[1].cols,
                matrices[2].cols,
                is_float,
                0,
                0,
                matrices[1].rows,
                BLOCK_SIZE
        );

        // D = A × T (m×k × k×n = m×n)
        multiply_matrices(
                matrices[0].data,
                temp[0].data,
                result->data,
                matrices[0].rows,
                matrices[0].cols,
                temp[0].cols,
                is_float,
                0,
                0,
                matrices[0].rows,
                BLOCK_SIZE
        );
    }
}

// matrix multiplication with 2D block partitioning in ikj order by 2-layer loop unrolling with an unrolling factor of 4
void multiply_matrices(void* A, void* B, void* C, int m, int k, int n, int is_float, int is_par, int start_row, int end_row, int block_size) {

    if (start_row < 0 || end_row > m || start_row >= end_row) {
        fprintf(stderr, "Error: Invalid row bounds for matrix multiplication.\n");
        return;
    }

    if (is_float) {
        float* A_float = (float*)A;
        float* B_float = (float*)B;
        float* C_float = (float*)C;

        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < n; j++) {
                C_float[i * n + j] = 0.0f;
            }
        }

        if (is_par) {
            // 2D block partitioning for parallel computation
            for (int i = start_row; i < end_row; i += block_size) { // row block of A
                for (int kk = 0; kk < k; kk += block_size) { // inner block
                    for (int j = 0; j < n; j += block_size) { // column block of B
                        for (int ii = i; ii < i + block_size && ii < end_row; ii++) {
                            // Unroll loop k with unrolling factor of 4
                            for (int kkk = kk; kkk < kk + block_size && kkk < k; kkk += 4) {
                                float A_result_1 = A_float[ii * k + kkk];
                                float A_result_2 = (kkk + 1 < k) ? A_float[ii * k + kkk + 1] : 0.0f;
                                float A_result_3 = (kkk + 2 < k) ? A_float[ii * k + kkk + 2] : 0.0f;
                                float A_result_4 = (kkk + 3 < k) ? A_float[ii * k + kkk + 3] : 0.0f;
                                // Unroll loop j with unrolling factor of 4
                                for (int jj = j; jj < j + block_size && jj < n; jj += 4) {
                                    if (jj < n) C_float[ii * n + jj] += A_result_1 * B_float[kkk * n + jj];
                                    if (jj + 1 < n) C_float[ii * n + jj + 1] += A_result_1 * B_float[kkk * n + jj + 1];
                                    if (jj + 2 < n) C_float[ii * n + jj + 2] += A_result_1 * B_float[kkk * n + jj + 2];
                                    if (jj + 3 < n) C_float[ii * n + jj + 3] += A_result_1 * B_float[kkk * n + jj + 3];

                                    if (jj < n) C_float[ii * n + jj] += A_result_2 * B_float[(kkk + 1) * n + jj];
                                    if (jj + 1 < n) C_float[ii * n + jj + 1] += A_result_2 * B_float[(kkk + 1) * n + jj + 1];
                                    if (jj + 2 < n) C_float[ii * n + jj + 2] += A_result_2 * B_float[(kkk + 1) * n + jj + 2];
                                    if (jj + 3 < n) C_float[ii * n + jj + 3] += A_result_2 * B_float[(kkk + 1) * n + jj + 3];

                                    if (jj < n) C_float[ii * n + jj] += A_result_3 * B_float[(kkk + 2) * n + jj];
                                    if (jj + 1 < n) C_float[ii * n + jj + 1] += A_result_3 * B_float[(kkk + 2) * n + jj + 1];
                                    if (jj + 2 < n) C_float[ii * n + jj + 2] += A_result_3 * B_float[(kkk + 2) * n + jj + 2];
                                    if (jj + 3 < n) C_float[ii * n + jj + 3] += A_result_3 * B_float[(kkk + 2) * n + jj + 3];

                                    if (jj < n) C_float[ii * n + jj] += A_result_4 * B_float[(kkk + 3) * n + jj];
                                    if (jj + 1 < n) C_float[ii * n + jj + 1] += A_result_4 * B_float[(kkk + 3) * n + jj + 1];
                                    if (jj + 2 < n) C_float[ii * n + jj + 2] += A_result_4 * B_float[(kkk + 3) * n + jj + 2];
                                    if (jj + 3 < n) C_float[ii * n + jj + 3] += A_result_4 * B_float[(kkk + 3) * n + jj + 3];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for (int i = start_row; i < end_row; i++) {
                for (int kk = 0; kk < k; kk++) {
                    for (int j = 0; j < n; j++) {
                        C_float[i * n + j] += A_float[i * k + kk] * B_float[kk * n + j];
                    }
                }
            }
        }
    } else {
        double* A_double = (double*)A;
        double* B_double = (double*)B;
        double* C_double = (double*)C;

        for (int i = start_row; i < end_row; i++) {
            for (int j = 0; j < n; j++) {
                C_double[i * n + j] = 0.0;
            }
        }

        if (is_par) {
            // 2D block partitioning for parallel computation
            for (int i = start_row; i < end_row; i += block_size) { // row block of A
                for (int kk = 0; kk < k; kk += block_size) { // inner block
                    for (int j = 0; j < n; j += block_size) { // column block of B
                        for (int ii = i; ii < i + block_size && ii < end_row; ii++) {
                            // Unroll loop k with unrolling factor of 4
                            for (int kkk = kk; kkk < kk + block_size && kkk < k; kkk += 4) {
                                double A_result_1 = A_double[ii * k + kkk];
                                double A_result_2 = (kkk + 1 < k) ? A_double[ii * k + kkk + 1] : 0.0;
                                double A_result_3 = (kkk + 2 < k) ? A_double[ii * k + kkk + 2] : 0.0;
                                double A_result_4 = (kkk + 3 < k) ? A_double[ii * k + kkk + 3] : 0.0;
                                // Unroll loop j with unrolling factor of 4
                                for (int jj = j; jj < j + block_size && jj < n; jj += 4) {
                                    if (jj < n) C_double[ii * n + jj] += A_result_1 * B_double[kkk * n + jj];
                                    if (jj + 1 < n) C_double[ii * n + jj + 1] += A_result_1 * B_double[kkk * n + jj + 1];
                                    if (jj + 2 < n) C_double[ii * n + jj + 2] += A_result_1 * B_double[kkk * n + jj + 2];
                                    if (jj + 3 < n) C_double[ii * n + jj + 3] += A_result_1 * B_double[kkk * n + jj + 3];

                                    if (jj < n) C_double[ii * n + jj] += A_result_2 * B_double[(kkk + 1) * n + jj];
                                    if (jj + 1 < n) C_double[ii * n + jj + 1] += A_result_2 * B_double[(kkk + 1) * n + jj + 1];
                                    if (jj + 2 < n) C_double[ii * n + jj + 2] += A_result_2 * B_double[(kkk + 1) * n + jj + 2];
                                    if (jj + 3 < n) C_double[ii * n + jj + 3] += A_result_2 * B_double[(kkk + 1) * n + jj + 3];

                                    if (jj < n) C_double[ii * n + jj] += A_result_3 * B_double[(kkk + 2) * n + jj];
                                    if (jj + 1 < n) C_double[ii * n + jj + 1] += A_result_3 * B_double[(kkk + 2) * n + jj + 1];
                                    if (jj + 2 < n) C_double[ii * n + jj + 2] += A_result_3 * B_double[(kkk + 2) * n + jj + 2];
                                    if (jj + 3 < n) C_double[ii * n + jj + 3] += A_result_3 * B_double[(kkk + 2) * n + jj + 3];

                                    if (jj < n) C_double[ii * n + jj] += A_result_4 * B_double[(kkk + 3) * n + jj];
                                    if (jj + 1 < n) C_double[ii * n + jj + 1] += A_result_4 * B_double[(kkk + 3) * n + jj + 1];
                                    if (jj + 2 < n) C_double[ii * n + jj + 2] += A_result_4 * B_double[(kkk + 3) * n + jj + 2];
                                    if (jj + 3 < n) C_double[ii * n + jj + 3] += A_result_4 * B_double[(kkk + 3) * n + jj + 3];
                                }
                            }
                        }
                    }
                }
            }
        } else {
            for (int i = start_row; i < end_row; i++) {
                for (int kk = 0; kk < k; kk++) {
                    for (int j = 0; j < n; j++) {
                        C_double[i * n + j] += A_double[i * k + kk] * B_double[kk * n + j];
                    }
                }
            }
        }
    }
}

void initialize_matrix(matrix_t* matrix, int is_float) {
    int size = matrix->rows * matrix->cols;
    if (is_float) {
        float* data = (float*)matrix->data;
        for (int i = 0; i < size; i++) {
            data[i] = (float)rand() / RAND_MAX;
        }
    } else {
        double* data = (double*)matrix->data;
        for (int i = 0; i < size; i++) {
            data[i] = (double)rand() / RAND_MAX;
        }
    }
}

// compare results between parallel and sequential computations to verify correctness
int compare_matrices(void* num_1, void* num_2, int rows, int cols, int is_float) {
    if (is_float) {
        float* float_1 = (float*)num_1;
        float* float_2 = (float*)num_2;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                if (fabsf(float_1[idx] - float_2[idx]) > 1e-7f) {
                    printf("\nMismatch at D(%d, %d): Sequential: %.7f      Parallel: %.7f\n", i, j, float_1[idx], float_2[idx]);
                    return 0;
                }
            }
        }
    } else {
        double* double_1 = (double*)num_1;
        double* double_2 = (double*)num_2;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int idx = i * cols + j;
                if (fabs(double_1[idx] - double_2[idx]) > 1e-15) {
                    printf("\nMismatch at D(%d, %d): Sequential: %.15lf      Parallel: %.15lf\n", i, j, double_1[idx], double_2[idx]);
                    return 0;
                }
            }
        }
    }
    return 1;
}
