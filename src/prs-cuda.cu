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
        
        /* rastrigin */

        double fitness = 10.0 * dim;
        for (uint32_t i = 0; i < dim; i++) {
                double solution_component = (x[i] / 90.0) * (upper[i] - lower[i]) + lower[i];
                // printf("angle %f -> solution: %f\n", x[i], solution_component);
                fitness += solution_component * solution_component - 10.0 * cos(2 * M_PI * solution_component);
        }
        return fitness;

        /* ackley */

        // double sum_sq = 0.0;
        // double sum_cos = 0.0;

        // for (uint32_t i = 0; i < dim; i++) {
        //         double solution_component = (x[i] / 90.0) * (upper[i] - lower[i]) + lower[i];
        //         sum_sq += solution_component * solution_component;
        //         sum_cos += cos(2 * M_PI * solution_component);
        // }

        // double term1 = -20.0 * __expf(-0.2 * sqrt(sum_sq / dim));
        // double term2 = -__expf(sum_cos / dim);

        // return term1 + term2 + 20.0 + __expf(1);

        /* sphere */

        // double fitness = 0.0;

        // for (uint32_t i = 0; i < dim; i++) {
        //         double solution_component = (x[i] / 90.0) * (upper[i] - lower[i]) + lower[i];
        //         fitness += solution_component * solution_component;
        // }

        // return fitness;

        /* rosenbrock */

        // double fitness = 0.0;

        // for (uint32_t i = 0; i < dim - 1; i++) {
        //         double xi = x[i];
        //         double xi1 = x[i + 1];
        //         double xi_component = (xi / 90.0) * (upper[i] - lower[i]) + lower[i];
        //         double xi1_component = (xi1 / 90.0) * (upper[i + 1] - lower[i + 1]) + lower[i + 1];
        //         fitness += 100.0 * (xi1_component - xi_component * xi_component) * (xi1_component - xi_component * xi_component) + (1.0 - xi_component) * (1.0 - xi_component);
        // }

        // return fitness;

        /* styblinski-tang */

        // double fitness = 0.0;

        // for (uint32_t i = 0; i < dim; i++) {
        //         double xi_comp = (x[i] / 90.0) * (upper[i] - lower[i]) + lower[i];
        //         fitness += (xi_comp * xi_comp * xi_comp * xi_comp) - (16.0 * xi_comp * xi_comp) + (5.0 * xi_comp);
        // }

        // fitness /= 2;
        // return fitness;

        /* griewank */

        // double sum = 0.0;
        // double product = 1.0;

        // for (uint32_t i = 0; i < dim; i++) {
        //         double xi = x[i];
        //         double xi_component = (xi / 90.0) * (upper[i] - lower[i]) + lower[i];
        //         sum += xi_component * xi_component;
        //         product *= cos(xi_component / sqrtf(i + 1));
        // }

        // return 1.0 + sum / 4000.0 - product;
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

__device__ uint8_t gen_random_unsigned_kernel(curandStatePhilox4_32_10_t* state, uint8_t min, uint8_t max) {
        if (min > max) {
                uint8_t temp = min;
                min = max;
                max = temp;
        }

        uint8_t range = max - min + 1;
        uint8_t rand_val = curand(state);

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

__device__ double prs_angle_to_solution_kernel(double* solution, double* lowerbound, double* upperbound) {
        return (*solution / 90.0) * (upperbound[0] - lowerbound[0]) + lowerbound[0];
}

__device__ double prs_solution_to_angle_kernel(double* solution, double* lowerbound, double* upperbound) {
        return (*solution - lowerbound[0]) / (upperbound[0] - lowerbound[0]) * 90.0;
}

/* CUDA kernels */

__global__ void init_states_kernel(curandStatePhilox4_32_10_t *states, unsigned long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed << 7, idx, idx * 1024, &states[idx]);
}

__global__ void prs_init_incident_angles_kernel(double* population, const uint32_t population_size, const uint32_t dim, curandStatePhilox4_32_10_t* states) {
        int solution = blockIdx.x * blockDim.x + threadIdx.x;
        if (solution >= (int)population_size) return;

        double* solution_data = &population[solution * dim];    // start index of solution
        
        curandStatePhilox4_32_10_t localState = states[solution];
        for (uint32_t i = 0; i < dim; i++) {
                solution_data[i] = (double)gen_random_unsigned_kernel(&localState, 0, 90);
        }

        states[solution] = localState;
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
        // printf("solution %d fitness: %f\n", solution, fitness);
        return;
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

__global__ void prs_update_emergent_incident_angles_kernel(double* emergent_angles, double* incident_angles, const double delta_t, const double prism_angle, double refractive_index, double* lower, double* upper, const uint32_t population_size, const uint32_t dim, curandStatePhilox4_32_10_t* states) {
        uint32_t total_threads = population_size * dim;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= (int)total_threads) return;

        curandStatePhilox4_32_10_t localState = states[i];

        double d_prism_angle = prism_angle;
        double incident_sol = prs_angle_to_solution_kernel(&incident_angles[i], lower, upper);
        double prism_sol = prs_angle_to_solution_kernel(&d_prism_angle, lower, upper);
        double emergent_sol = delta_t - incident_sol + prism_sol;
        emergent_angles[i] = prs_solution_to_angle_kernel(&emergent_sol, lower, upper);

        double r1 = 0.3 + 0.7 * gen_random_double_kernel(&localState);

        double emergent_angle_rad = DEG2RAD(emergent_angles[i]);
        double prism_angle_rad = DEG2RAD(prism_angle);
        double sin_e = sin(emergent_angle_rad);
        double inside = refractive_index * refractive_index - sin_e * sin_e;
        inside = fmax(0.0, inside);
        double val = -sin_e
                * cos(prism_angle_rad)
                + r1 * sin(prism_angle_rad)
                * sqrt(inside);
        val = fmin(0.999, fmax(-0.999, val));

        double incident_angle_rad = asin(val);
        incident_angles[i] = fmin(90.0, fmax(10.0, RAD2DEG(incident_angle_rad)));
        
        states[i] = localState;
        return;
}

/* prs core functions */

double prs_init_prism_angle() {
        return (double)gen_random_unsigned(15, 75);
}

double prs_angle_to_solution(double* angle, double* lowerbound, double* upperbound) {
        return (*angle / 90.0) * (upperbound[0] - lowerbound[0]) + lowerbound[0];
}

double prs_solution_to_angle(double* solution, double* lowerbound, double* upperbound) {
        return (*solution - lowerbound[0]) / (upperbound[0] - lowerbound[0]) * 90.0;
}

double prs_get_refractive_index(double* prism_angle, double* delta, const uint32_t dim) {
        double prism_angle_rad = DEG2RAD(*prism_angle);
        double denom = sin(prism_angle_rad / 2.0);
        denom = fmax(1e-6, denom);
        double delta_angle = *delta / (dim * 40 + 5) * 90.0;
        double inside_num = (*prism_angle + delta_angle) / 2.0;
        inside_num = DEG2RAD(inside_num);
        return sin(inside_num) / denom;
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

        int block_size = 1024;
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

                // printf("delta_t sum: %f\n", h_delta_t);
                h_delta_t = fmax(1e-6, h_delta_t / params->population_size);
                // printf("delta_t avg: %f\n", h_delta_t);

                /* calculate refractive index*/

                // printf("prism angle %f\n", h_prism_angle);
                double refractive_index = prs_get_refractive_index(&h_prism_angle, &h_delta_t, params->dim);
                // printf("refractive index: %f\n", refractive_index);

                /* calculate emergent and update incident angles */

                grid_size = (params->population_size * params->dim + block_size - 1) / block_size;
                prs_update_emergent_incident_angles_kernel<<<grid_size, block_size>>> (
                        d_emergent_angles,
                        d_incident_angles,
                        h_delta_t,
                        h_prism_angle,
                        refractive_index,
                        d_lowerbound,
                        d_upperbound,
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

                double local_best_score = 0.0;
                cudaMemcpy(&local_best_score, d_best_score, sizeof(double), cudaMemcpyDeviceToHost);
                if (local_best_score < *h_best_score) {
                        *h_best_score = local_best_score;
                        cudaMemcpy(h_best_solution, d_incident_angles, params->dim * sizeof(double), cudaMemcpyDeviceToHost);
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
                }
                cudaFree(d_best_score_index);

        }

        /* update best score and solution */

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
        params.dim = 4;
        params.max_iter = 10000;
        params.alpha = 0.009;
        params.population_size = 1024;
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
