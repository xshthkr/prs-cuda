/* custom header */
#include <prs.h>
#include <utils.h>

/* standard headers */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>

int main() {

        prs_params_t params;
        params.dim = 4;
        params.max_iter = 6000;
        params.alpha = 0.009;
        params.population_size = 1000;
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

/*

Dim             Population Size         Max Iterations
2–5             100–300                 500–1000
10              300–600                 1000–3000
20              600–1000                3000–5000
30+             1000–2000               5000–10000+

*/