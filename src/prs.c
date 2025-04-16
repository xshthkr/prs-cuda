#include <prs.h>

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

void prs_optimizer(const prs_params_t* params, double* lowerbound, double* upperbound, double* best_solution, double* best_score) {
        
        // wtf is this

        // initialize population
        double** population = prs_init_population(params, lowerbound, upperbound);

        // initialize prism angle 
        double prism_angle = prs_init_prism_angle(params, lowerbound, upperbound);

        // for every iteration
        for (uint32_t iter = 0; iter < params->max_iter; iter++) {

                // for every solution in the population
                for (uint32_t i = 0; i < params->population_size; i++) {

                        // get fitness (delta)
                        // if delta < best score
                        // update best score
                        // update best solution
                }

                // calculate refractive index

                // for every solution in the population
                for (uint32_t i = 0; i < params->population_size; i++) {

                        // for every dimension of the solution
                        for (uint32_t j = 0; j < params->dim; j++) {
                                // update emergent angle component
                                // generate random number  [-1, 1]
                                // update incident angle component
                                // E(t,j) = delta(t) - i(t,j) + A(t)
                                // i and E are updated componentially
                                // ensure i(t,j) is within bounds
                        }
                }

                // update prism angle
        }
        
        return;
}

double** prs_init_population(const prs_params_t* params, double* lowerbound, double* upperbound) {
        double** population = (double**)malloc(params->population_size * sizeof(double*));
        assert(population != NULL);

        for(uint32_t i = 0; i < params->population_size; i++) {
                population[i] = (double*)malloc(params->dim * sizeof(double));
                assert(population[i] != NULL);
                for (uint32_t j = 0; j < params->dim; j++) {
                        // population[i][j] = lowerbound[j] + (upperbound[j] - lowerbound[j]) * random(0, 90)
                }
        }

        return population;
}

double prs_init_prism_angle(const prs_params_t* params, double* lowerbound, double* upperbound) {
        // return max(lowerbound) + (min(upperbound) - max(lowerbound)) * random(0, 90)
}

void prs_print_params(const prs_params_t* params) {
        printf("Parameters:\n");
        printf("  Dimension: %u\n", params->dim);
        printf("  Max Iterations: %u\n", params->max_iter);
        printf("  Population size: %u\n", params->population_size);
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