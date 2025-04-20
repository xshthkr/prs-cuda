#include <cuda.h>
#include <curand_kernel.h>

#include <math.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

/* macros */

#define M_PI 3.14159265358979323846
#define DBL_MAX 1.7976931348623158e+308
#define DEG2RAD(x) ((x) * M_PI / 180.0)
#define RAD2DEG(x) ((x) * 180.0 / M_PI)

/* structs */

typedef struct {
        uint32_t                dim;
        uint32_t                max_iter;
        uint32_t                population_size;
        float                   alpha;
} prs_params_t;

/* util functions */

__device__ double eval_fitness_kernel(const double* x, const double* lower, const double* upper, const uint32_t dim) {
        double fitness = 10.0 * dim;
        for (uint32_t i = 0; i < dim; i++) {
                double solution_component = (x[i] / 90.0) * (upper[i] - lower[i]) + lower[i];
                fitness += solution_component * solution_component - 10.0 * cos(2 * M_PI * solution_component);
        }
        return fitness;
}

uint8_t gen_random_unsigned(uint8_t lower, uint8_t upper) {
        if (lower > upper) {
                uint8_t temp = lower;
                lower = upper;
                upper = temp;
        }

        uint8_t range = upper - lower + 1;
        uint8_t rand_val;
        int fd = open("/dev/urandom", O_RDONLY);
        if (fd < 0) {
                perror("Failed to open /dev/urandom");
                exit(EXIT_FAILURE);
        }

        uint8_t limit = UINT8_MAX - (UINT8_MAX % range);

        do {
                if (read(fd, &rand_val, sizeof(rand_val)) != sizeof(rand_val)) {
                perror("Failed to read random bytes");
                close(fd);
                exit(EXIT_FAILURE);
                }
        } while ((uint8_t)rand_val >= limit);

        close(fd);
        return lower + ((uint8_t)rand_val % range);
}

__device__ uint32_t gen_random_unsigned_kernel(curandStatePhilox4_32_10_t* state, uint32_t min, uint32_t max) {
        if (min > max) {
                uint32_t temp = min;
                min = max;
                max = temp;
        }

        uint32_t range = max - min + 1;
        uint32_t rand_val = curand(state);

        return min + (rand_val % range);
}

__device__ double gen_random_double_kernel(curandStatePhilox4_32_10_t* state) {
        return (curand_uniform_double(state) * 2.0) - 1.0;
}

void prs_print_params(const prs_params_t* params) {
        printf("Parameters:\n");
        printf("  Dimension: %u\n", params->dim);
        printf("  Max Iterations: %u\n", params->max_iter);
        printf("  Population size: %u\n", params->population_size);
        printf("  Alpha: %f\n", params->alpha);
        return;
}

void prs_print_solution(const prs_params_t* params, const double* solution, const double* score) {
        printf("Best solution:\n");
        for (uint32_t i = 0; i < params->dim; i++) {
                printf("  %f\n", solution[i]);
        }
        printf("Best score: %f\n", *score);
        return;
}

void getCudaStats() {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        printf("Number of CUDA devices: %d\n", deviceCount);
        
        for (int i = 0; i < deviceCount; i++) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                printf("Device %d: %s\n", i, prop.name);
                printf("  Total global memory: %zu bytes\n", prop.totalGlobalMem);
                printf("  Multiprocessor count: %d\n", prop.multiProcessorCount);
                printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        }
        
        return;
}

/* CUDA kernels */

__global__ void init_states_kernel(curandStatePhilox4_32_10_t *states, unsigned long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed ^ (idx * 7919) + clock64(), idx, idx * 17, &states[idx]);
}

__global__ void prs_init_incident_angles_kernel(double* population, const uint32_t population_size, const uint32_t dim, curandStatePhilox4_32_10_t* states) {
        int solution = blockIdx.x * blockDim.x + threadIdx.x;
        if (solution >= (int)population_size) return;

        double* solution_data = &population[solution * dim];    // start index of solution
        
        curandStatePhilox4_32_10_t localState = states[solution];
        for (uint32_t i = 0; i < dim; i++) {
                solution_data[i] = gen_random_unsigned_kernel(&localState, 0, 90);
        }

        return;
}

__global__ void prs_init_emergent_angles_kernel(double* population, const uint32_t population_size, const uint32_t dim) {
        int individual_angle = blockIdx.x * blockDim.x + threadIdx.x;
        if (individual_angle >= (int)population_size) return;

        double* individual_angle_data = &population[individual_angle * dim];

        for (uint32_t i = 0; i < dim; i++) {
                individual_angle_data[i] = 0.0;
        }

        return;
}

__global__ void prs_calculate_fitness_kernel(double* population, double* fitness_values, double* lower, double* upper, const uint32_t population_size, const uint32_t dim) {
        int solution = blockIdx.x * blockDim.x + threadIdx.x;
        if (solution >= (int)population_size) return;

        double* solution_data = &population[solution * dim];    // start index of solution
        double fitness = eval_fitness_kernel(solution_data, lower, upper, dim);
        fitness_values[solution] = fitness;
}

__global__ void prs_reduce_fitness_sum_block_kernel(
        const double* __restrict__ fitness_values,
        double* __restrict__ block_sums,
        uint32_t population_size
        ) {

        extern __shared__ double shared[];
        uint32_t tid = threadIdx.x;
        uint32_t global_idx = blockIdx.x * blockDim.x + tid;

        shared[tid] = (global_idx < population_size) ? fitness_values[global_idx] : 0.0;
        __syncthreads();

        // Reduce in shared memory
        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s)
                shared[tid] += shared[tid + s];
                __syncthreads();
        }

        if (tid == 0)
                block_sums[blockIdx.x] = shared[0];
}

__global__ void prs_reduce_fitness_sum_final_kernel(
        const double* __restrict__ block_sums,
        double* __restrict__ final_sum,
        uint32_t num_blocks
        ) {

        extern __shared__ double shared[];
        uint32_t tid = threadIdx.x;

        shared[tid] = (tid < num_blocks) ? block_sums[tid] : 0.0;
        __syncthreads();

        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s)
                shared[tid] += shared[tid + s];
                __syncthreads();
        }

        if (tid == 0)
                *final_sum = shared[0];
        }

__global__ void prs_reduce_fitness_min_block_kernel(
                const double* __restrict__ fitness_values,
                double* __restrict__ block_mins,
                uint32_t* __restrict__ block_indices,
                uint32_t population_size
        ) {

        extern __shared__ double shared[];
        double* shared_vals = shared;
        uint32_t* shared_idx = (uint32_t*)&shared[blockDim.x];

        uint32_t tid = threadIdx.x;
        uint32_t global_idx = blockIdx.x * blockDim.x + tid;

        double val = (global_idx < population_size) ? fitness_values[global_idx] : DBL_MAX;
        shared_vals[tid] = val;
        shared_idx[tid] = global_idx;
        __syncthreads();

        for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s && shared_vals[tid] > shared_vals[tid + s]) {
                shared_vals[tid] = shared_vals[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
                }
                __syncthreads();
        }

        if (tid == 0) {
                block_mins[blockIdx.x] = shared_vals[0];
                block_indices[blockIdx.x] = shared_idx[0];
        }
}

__global__ void prs_reduce_fitness_min_final_kernel(
                const double* __restrict__ block_mins,
                const uint32_t* __restrict__ block_indices,
                double* __restrict__ final_min,
                uint32_t* __restrict__ final_index,
                uint32_t num_blocks
        ) {

        extern __shared__ double shared[];
        double* shared_vals = shared;
        uint32_t* shared_idx = (uint32_t*)&shared[num_blocks];

        uint32_t tid = threadIdx.x;

        shared_vals[tid] = (tid < num_blocks) ? block_mins[tid] : DBL_MAX;
        shared_idx[tid] = (tid < num_blocks) ? block_indices[tid] : 0;
        __syncthreads();

        for (uint32_t s = num_blocks / 2; s > 0; s >>= 1) {
                if (tid < s && shared_vals[tid] > shared_vals[tid + s]) {
                shared_vals[tid] = shared_vals[tid + s];
                shared_idx[tid] = shared_idx[tid + s];
                }
                __syncthreads();
        }

        if (tid == 0) {
                *final_min = shared_vals[0];
                *final_index = shared_idx[0];
        }
}

__global__ void prs_copy_best_solution_kernel(double* incident_angles, double* best_solution, const uint32_t* index, const double* lower, const double* upper, const uint32_t dim) {
        int component = blockIdx.x * blockDim.x + threadIdx.x;
        if (component >= (int)dim) return;

        double* incident_angle_data = &incident_angles[dim * (*index)];
        best_solution[component] = (incident_angle_data[component] / 90.0) * (upper[component] - lower[component]) + lower[component];
        
        return;
}

__global__ void prs_update_emergent_incident_angles_kernel(double* emergent_angles, double* incident_angles, const double delta_t, const double prism_angle, double refractive_index, const uint32_t population_size, const uint32_t dim, curandStatePhilox4_32_10_t* states) {
        uint32_t total_threads = population_size * dim;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= (int)total_threads) return;

        curandStatePhilox4_32_10_t localState = states[i];
        emergent_angles[i] = delta_t - incident_angles[i] + prism_angle;
        double r1 = gen_random_double_kernel(&localState);

        double emergent_angle_rad = DEG2RAD(emergent_angles[i]);
        double prism_angle_rad = DEG2RAD(prism_angle);
        double sin_e = sin(emergent_angle_rad);
        double inside = refractive_index * refractive_index - sin_e * sin_e;
        inside = fmax(0.0, inside);
        double val = -sin_e
                * cos(prism_angle_rad)
                + r1 * sin(prism_angle_rad)
                * sqrt(inside);
        val = fmin(1.0, fmax(-1.0, val));

        double incident_angle_rad = asin(val);
        incident_angles[i] = RAD2DEG(incident_angle_rad);
}

/* prs core functions */

double prs_init_prism_angle() {
        return (double)gen_random_unsigned(15, 75);
}

double prs_get_refractive_index(double* prism_angle, double* delta) {
        double denom = sin(*prism_angle / 2.0);
        denom = fmax(1e-6, denom);
        return sin((*prism_angle + *delta) / 2.0) / denom;
}

void prs_optimizer(const prs_params_t* params, double* lowerbound, 
        double* upperbound, double* h_best_solution, double* h_best_score) {

        /* data structures */

        double* d_incident_angles;      // flattened 1D array
        double* d_emergent_angles;      // flattened 1D array
        double* d_prism_angle;          // copied from cpu
        double* d_best_solution;
        double* d_best_score;
        double* d_fitness_values;
        double* d_lowerbound;           // copied from cpu
        double* d_upperbound;           // copied from cpu

        curandStatePhilox4_32_10_t* d_states;

        /* allocate */

        int block_size = 256;
        int grid_size = (params->population_size + block_size - 1) / block_size;

        cudaMalloc((void**)&d_incident_angles, params->population_size * params->dim * sizeof(double));
        cudaMalloc((void**)&d_emergent_angles, params->population_size * params->dim * sizeof(double));
        cudaMalloc((void**)&d_prism_angle, sizeof(double));
        cudaMalloc((void**)&d_best_solution, params->dim * sizeof(double));
        cudaMalloc((void**)&d_best_score, sizeof(double));
        cudaMalloc((void**)&d_fitness_values, params->population_size * sizeof(double));
        cudaMalloc((void**)&d_lowerbound, params->dim * sizeof(double));
        cudaMalloc((void**)&d_upperbound, params->dim * sizeof(double));
        cudaMalloc((void**)&d_states, block_size * grid_size * sizeof(curandStatePhilox4_32_10_t));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error [Initial mem allocation]: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }


        /* initialize */

        grid_size = (params->population_size * params->dim + block_size - 1) / block_size;
        init_states_kernel<<<grid_size, block_size>>> (d_states, time(NULL));
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error [Init states for RNG]: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

        grid_size = (params->population_size + block_size - 1) / block_size;        
        prs_init_incident_angles_kernel<<<grid_size, block_size>>> (d_incident_angles, params->population_size, params->dim, d_states);
        prs_init_emergent_angles_kernel<<<grid_size, block_size>>> (d_emergent_angles, params->population_size, params->dim);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error [Init incident/emergent angles]: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

        *h_best_score = DBL_MAX;
        cudaMemcpy(d_best_score, h_best_score, sizeof(double), cudaMemcpyHostToDevice);

        double h_prism_angle = prs_init_prism_angle();
        cudaMemcpy(d_prism_angle, &h_prism_angle, sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_lowerbound, lowerbound, params->dim * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_upperbound, upperbound, params->dim * sizeof(double), cudaMemcpyHostToDevice);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error [Initial memcpy to device]: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

        /* calculate delta of every solution */

        prs_calculate_fitness_kernel<<<grid_size, block_size>>> (
                d_incident_angles, 
                d_fitness_values, 
                d_lowerbound,
                d_upperbound,
                params->population_size,
                params->dim
        );
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error [initial fitness calculation]: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

        /* main loop */

        for (uint32_t iter = 0; iter < params->max_iter; iter++) {


                /* reduce to average delta_t */

                double h_delta_t = 0.0;

                double* d_delta_t;
                double* d_block_sums;
                cudaMalloc((void**)&d_block_sums, grid_size * sizeof(double));
                cudaMalloc((void**)&d_delta_t, sizeof(double));
                cudaMemcpy(d_delta_t, &h_delta_t, sizeof(double), cudaMemcpyHostToDevice);
                prs_reduce_fitness_sum_block_kernel<<<grid_size, block_size, block_size * sizeof(double)>>>(d_fitness_values, d_block_sums, params->population_size);
                prs_reduce_fitness_sum_final_kernel<<<1, grid_size, grid_size * sizeof(double)>>>(d_block_sums, d_delta_t, grid_size);                
                cudaDeviceSynchronize();
                cudaMemcpy(&h_delta_t, d_delta_t, sizeof(double), cudaMemcpyDeviceToHost);
                cudaFree(d_delta_t);
                cudaFree(d_block_sums);

                cudaDeviceSynchronize();
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                        fprintf(stderr, "CUDA Error [Fitness sum reduction]: %s\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                }

                h_delta_t = fmax(1e-6, h_delta_t / params->population_size);

                /* calculate refractive index*/

                double refractive_index = prs_get_refractive_index(&h_prism_angle, &h_delta_t);

                /* calculate emergent angles */

                grid_size = (params->population_size * params->dim + block_size - 1) / block_size;
                prs_update_emergent_incident_angles_kernel<<<grid_size, block_size>>> (
                        d_emergent_angles,
                        d_incident_angles,
                        h_delta_t,
                        h_prism_angle,
                        refractive_index,
                        params->population_size,
                        params->dim,
                        d_states
                );
                cudaDeviceSynchronize();
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                        fprintf(stderr, "CUDA Error [Update emergent/incident angles]: %s\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                }

                /* update prism angle */

                h_prism_angle *= exp(-params->alpha * (double)iter / (double)params->max_iter);

                /* find solution with best score */

                grid_size = (params->population_size + block_size - 1) / block_size;          
                prs_calculate_fitness_kernel<<<grid_size, block_size>>> (
                        d_incident_angles,
                        d_fitness_values,
                        d_lowerbound,
                        d_upperbound,
                        params->population_size,
                        params->dim
                );
                cudaDeviceSynchronize();
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                        fprintf(stderr, "CUDA Error [Loop calculate fitness]: %s\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                }

                uint32_t* d_best_score_index;
                double* d_block_mins;
                uint32_t* d_block_indices;
                cudaMalloc(&d_block_mins, grid_size * sizeof(double));
                cudaMalloc(&d_block_indices, grid_size * sizeof(uint32_t));
                cudaMalloc(&d_best_score_index, sizeof(uint32_t));
                prs_reduce_fitness_min_block_kernel<<<grid_size, block_size, block_size * (sizeof(double) + sizeof(uint32_t))>>>(d_fitness_values, d_block_mins, d_block_indices, params->population_size);
                prs_reduce_fitness_min_final_kernel<<<1, grid_size, grid_size * (sizeof(double) + sizeof(uint32_t))>>>(d_block_mins, d_block_indices, d_best_score, d_best_score_index, grid_size);
                cudaDeviceSynchronize();
                cudaFree(d_block_mins);
                cudaFree(d_block_indices);
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                        fprintf(stderr, "CUDA Error [Fitness max reduction]: %s\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                }

                prs_copy_best_solution_kernel<<<1, params->dim>>> (
                        d_incident_angles,
                        d_best_solution,
                        d_best_score_index,
                        d_lowerbound,
                        d_upperbound,
                        params->dim
                );
                cudaDeviceSynchronize();
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                        fprintf(stderr, "CUDA Error [Get best solution]: %s\n", cudaGetErrorString(err));
                        exit(EXIT_FAILURE);
                }
                cudaFree(d_best_score_index);

        }

        /* update best score and solution */

        cudaMemcpy(h_best_score, d_best_score, sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_best_solution, d_best_solution, params->dim * sizeof(double), cudaMemcpyDeviceToHost);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error [Final memcpy to host]: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }

        /* free resources */

        cudaFree(d_states);
        cudaFree(d_best_score);
        cudaFree(d_best_solution);
        cudaFree(d_incident_angles);
        cudaFree(d_emergent_angles);
        cudaFree(d_fitness_values);
        cudaFree(d_prism_angle);
        cudaFree(d_lowerbound);
        cudaFree(d_upperbound);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
                fprintf(stderr, "CUDA Error [Free mem]: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
        }
        
        return;
}

/* entry */

int main() {

        getCudaStats();

        prs_params_t params;
        params.dim = 3;
        params.max_iter = 1000;
        params.alpha = 0.009;
        params.population_size = 100;
        double* lowerbound = (double*)malloc(params.dim * sizeof(double));
        double* upperbound = (double*)malloc(params.dim * sizeof(double));
        for (uint32_t i = 0; i < params.dim; i++) {
                lowerbound[i] = -5.12;
                upperbound[i] = 5.12;
        }

        double *best_solution = (double*)malloc(params.dim * sizeof(double));
        double *best_score = (double*)malloc(sizeof(double));

        assert(best_solution != NULL);
        assert(best_score != NULL);

        clock_t start_time = clock();
        prs_optimizer(&params, lowerbound, upperbound, best_solution, best_score);
        clock_t end_time = clock();

        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        prs_print_params(&params);
        prs_print_solution(&params, best_solution, best_score);
        printf("Elapsed time: %f seconds\n", elapsed_time);

        free(best_solution);
        free(best_score);
        free(lowerbound);
        free(upperbound);        

        return 0;
}
