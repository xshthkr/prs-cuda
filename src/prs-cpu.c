#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

typedef struct {
        uint32_t dim;
        uint32_t max_iter;
        uint32_t population_size;
} prs_params_t;

void prs_print_params(const prs_params_t* params);

void prs_print_solution(
        const prs_params_t*     params,
        const double*           solution,
        const double*           score
);

double eval_fitness(
        const double*           solution,
        const uint32_t*         dim
);

void prs_optimizer(
        const prs_params_t*     params,
        double*                 best_solution,
        double*                 best_score
);

int main() {

        prs_params_t params;
        params.dim = 10;
        params.max_iter = 1000;
        params.population_size = 50;

        double *best_solution = (double*)malloc(params.dim * sizeof(double));
        double *best_score = (double*)malloc(sizeof(double));

        assert(best_solution != NULL);
        assert(best_score != NULL);

        memset(best_solution, 0, params.dim * sizeof(double));
        memset(best_score, 0, sizeof(double));

        clock_t start_time = clock();
        prs_optimizer(&params, best_solution, best_score);
        clock_t end_time = clock();

        double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

        printf("Elapsed time: %f seconds\n", elapsed_time);
        prs_print_params(&params);
        prs_print_solution(&params, best_solution, best_score);

        free(best_solution);
        free(best_score);

        return 0;
}

double eval_fitness(const double* x, const uint32_t* dim) {
        double fitness = 0.0;
        for (uint32_t i = 0; i < *dim; i++) {
                fitness += x[i] * x[i];
        }
        return fitness;
}

void prs_optimizer(const prs_params_t* params, double* best_solution, double* best_score) {
        
        // wtf is this

        for (uint32_t iter = 0; iter < params->max_iter; iter++) {
                for (uint32_t i = 0; i < params->population_size; i++) {
                        for (uint32_t j = 0; j < params->dim; j++) {
                                // get fitness
                                // if delta < best score
                                // update best score
                        }
                }
                // calculate refractive index
                for (uint32_t* i = 0; i < params->dim; i++) {
                        for (uint32_t j = 0; j < params->dim; j++) {
                                // update emergent angle
                                // generate random number  [-1, 1]
                                // update incident angle
                        }
                }
                // update prism angle
                // update best solution and position
        }
        
        return;
}

void prs_print_params(const prs_params_t* params) {
        printf("Parameters:\n");
        printf("  Dimension: %u\n", params->dim);
        printf("  Max Iterations: %u\n", params->max_iter);
        printf("  Population size: %u\n", params->population_size);
        return;
}

void prs_print_solutions(const prs_params_t* params, const double* solution, const double* score) {
        printf("Best solution:\n");
        for (uint32_t i = 0; i < params->dim; i++) {
                printf("  %f\n", solution[i]);
        }
        printf("Best score: %f\n", *score);
        return;
}