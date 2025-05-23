#include <prs.h>
#include <utils.h>

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* macros */
#define M_PI 3.14159265358979323846
#define DBL_MAX 1.7976931348623158e+308
#define DEG2RAD(x) ((x) * M_PI / 180.0)
#define RAD2DEG(x) ((x) * 180.0 / M_PI)

void prs_optimizer(const prs_params_t* params, double* lowerbound, 
        double* upperbound, double* best_solution, double* best_score) {
        
        // initialize population and prism angle
        double** incident_angles = prs_init_incidence_angles(params);
        double** emergent_angles = prs_init_emergent_angles(params);
        double prism_angle = prs_init_prism_angle();

        *best_score = DBL_MAX;
        double* solution = (double*)malloc(params->dim * sizeof(double));

        // for every iteration
        for (uint32_t iter = 0; iter < params->max_iter; iter++) {

                double delta_t = 0.0;

                // for every solution in the population
                for (uint32_t i = 0; i < params->population_size; i++) {

                        // get fitness (delta)
                        for (uint32_t j = 0; j < params->dim; j++) {
                                solution[j] = prs_angle_to_solution(
                                        &incident_angles[i][j], 
                                        &lowerbound[j], 
                                        &upperbound[j]
                                );
                        }
                        double delta = eval_fitness(solution, params->dim);
                        delta_t += delta;

                        // if delta < best score update best score and best solution
                        if (delta < *best_score) {
                                *best_score = delta;
                                for (uint32_t j = 0; j < params->dim; j++) {
                                        best_solution[j] = solution[j];
                                }
                        }
                }
                

                // calculate refractive index
                delta_t = fmax(1e-6, delta_t / params->population_size);
                double refractive_index = prs_get_refractive_index(&prism_angle, &delta_t, params->dim);

                // for every solution in the population
                for (uint32_t i = 0; i < params->population_size; i++) {

                        // for every dimension of the solution
                        for (uint32_t j = 0; j < params->dim; j++) {

                                // update emergent angle component
                                // E(t,j) = delta(t) - i(t,j) + A(t)
                                double incident_sol = prs_angle_to_solution(
                                        &incident_angles[i][j], 
                                        &lowerbound[j], 
                                        &upperbound[j]
                                );
                                double prism_sol = prs_angle_to_solution(
                                        &prism_angle, 
                                        &lowerbound[j], 
                                        &upperbound[j]
                                );
                                double emergent_sol = delta_t - incident_sol + prism_sol;
                                emergent_angles[i][j] = prs_solution_to_angle(
                                        &emergent_sol, 
                                        &lowerbound[j], 
                                        &upperbound[j]
                                );

                                // generate random number  [-1, 1]
                                double r1 = gen_random_double(-1.0, 1.0);

                                double emergent_angle_rad = DEG2RAD(emergent_angles[i][j]);
                                double prism_angle_rad = DEG2RAD(prism_angle);
                                double sin_e = sin(emergent_angle_rad);
                                double inside = refractive_index * refractive_index - sin_e * sin_e;
                                inside = fmax(0.0, inside);
                                double val = -sin_e 
                                        * cos(prism_angle_rad)
                                        + r1 * sin(prism_angle_rad)
                                        * sqrt(inside);
                                val = fmin(1.0, fmax(-1.0, val));

                                // update incident angle component
                                double incident_angle_rad = asin(val);

                                // incident_angles[i][j] = fmin(90.0, fmax(0.0, RAD2DEG(incident_angle_rad)));
                                incident_angles[i][j] = RAD2DEG(incident_angle_rad);
                        }
                }

                // update prism angle
                prism_angle *= exp(-params->alpha * (double)iter / (double)params->max_iter);

        }

        /* free resources */
        free(solution);
        for (uint32_t i = 0; i < params->population_size; i++) {
                free(incident_angles[i]);
                free(emergent_angles[i]);
        }
        free(incident_angles);
        free(emergent_angles);
        
        return;
}

double** prs_init_incidence_angles(const prs_params_t* params) {
        double** population = (double**)malloc(params->population_size * sizeof(double*));
        assert(population != NULL);

        for(uint32_t i = 0; i < params->population_size; i++) {
                population[i] = (double*)malloc(params->dim * sizeof(double));
                assert(population[i] != NULL);
                for (uint32_t j = 0; j < params->dim; j++) {
                        population[i][j] = (double)gen_random_unsigned(0, 90);
                }
        }

        return population;
}

double prs_init_prism_angle() {
        return (double)gen_random_unsigned(15, 75);
}

double** prs_init_emergent_angles(const prs_params_t* params) {
        double** emergent_angles = (double**)malloc(params->population_size * sizeof(double*));
        assert(emergent_angles != NULL);
        
        for (uint32_t i = 0; i < params->population_size; i++) {
                emergent_angles[i] = (double*)calloc(params->dim, sizeof(double));
                assert(emergent_angles[i] != NULL);
        }

        return emergent_angles;
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

double prs_angle_to_solution(double* angle, double* lowerbound, double* upperbound) {
        // double clamped_angle = fmin(90.0, fmax(0.0, *angle));
        // return (clamped_angle / 90.0) * (*upperbound - *lowerbound) + *lowerbound;
        return (*angle / 90.0) * (*upperbound - *lowerbound) + *lowerbound;
}

double prs_solution_to_angle(double* solution, double* lowerbound, double* upperbound) {
        // return fmin(90.0, fmax(0.0, (*solution - *lowerbound) / (*upperbound - *lowerbound) * 90.0));
        return (*solution - *lowerbound) / (*upperbound - *lowerbound) * 90.0;
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