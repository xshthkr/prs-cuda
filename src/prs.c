#include <prs.h>
#include <utils.h>

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>

void prs_optimizer(const prs_params_t* params, double* lowerbound, double* upperbound, double* best_solution, double* best_score) {
        
        // wtf is this

        // initialize population
        double** incident_angles = prs_init_population(params, lowerbound, upperbound);

        // initialize prism angle 
        double prism_angle = prs_init_prism_angle(params, lowerbound, upperbound);

        double** emergent_angles = prs_init_emergent_angles(params);

        // for every iteration
        for (uint32_t iter = 0; iter < params->max_iter; iter++) {

                double delta = 0.0;

                // for every solution in the population
                for (uint32_t i = 0; i < params->population_size; i++) {

                        // get fitness (delta)
                        delta = eval_fitness(incident_angles[i], params->dim);
                        // if delta < best score
                        // update best score
                        // update best solution
                        if (delta < *best_score) {
                                *best_score = delta;
                                for (uint32_t j = 0; j < params->dim; j++) {
                                        best_solution[j] = incident_angles[i][j];
                                }
                        }
                }

                // calculate refractive index
                double refractive_index = prs_get_refractive_index(&prism_angle, &delta);

                // for every solution in the population
                for (uint32_t i = 0; i < params->population_size; i++) {

                        // for every dimension of the solution
                        for (uint32_t j = 0; j < params->dim; j++) {

                                // i and E are updated componentially

                                // update emergent angle component
                                // E(t,j) = delta(t) - i(t,j) + A(t)
                                emergent_angles[i][j] = delta - incident_angles[i][j] + prism_angle;
                                // generate random number  [-1, 1]
                                double random_num = gen_random_double(-1, 1);
                                // update incident angle component
                                // incident_angles[i][j] = asin(-sin(emergent_angles[i][j]) * cos(prism_angle) + random_num * sin(prism_angle) * sqrt(pow(refractive_index, 2) - pow(sin(emergent_angles[i][j]), 2)));
                                incident_angles[i][j] += random_num * (emergent_angles[i][j] - incident_angles[i][j]);

                                // ensure i(t,j) is within bounds
                                incident_angles[i][j] = fmax(lowerbound[j], fmin(upperbound[j], incident_angles[i][j]));
                        }
                }

                // update prism angle
                double new_prism_angle = prism_angle * exp(-params->alpha * iter / params->max_iter);
                prism_angle = new_prism_angle;

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
                        population[i][j] = lowerbound[j] + (upperbound[j] - lowerbound[j]) * (double)gen_random_unsigned(0, 90);
                }
        }

        return population;
}

double prs_init_prism_angle(const prs_params_t* params, double* lowerbound, double* upperbound) {
        return max(lowerbound, params->dim) + (min(upperbound, params->dim) - max(lowerbound, params->dim)) * (double)gen_random_unsigned(0, 90);
}

double** prs_init_emergent_angles(const prs_params_t* params) {
        double** emergent_angles = (double**)malloc(params->population_size * sizeof(double*));
        assert(emergent_angles != NULL);
        
        for (uint32_t i = 0; i < params->population_size; i++) {
                emergent_angles[i] = (double*)malloc(params->dim * sizeof(double));
                assert(emergent_angles[i] != NULL);
                for (uint32_t j = 0; j < params->dim; j++) {
                        emergent_angles[i][j] = 0.0;
                }
        }

        return emergent_angles;
}

double prs_get_refractive_index(double* prism_angle, double* delta) {
        return (double) sin((*prism_angle + *delta) / 2) / sin(*prism_angle / 2);
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