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
        params.dim = 3;
        params.max_iter = 1000;
        params.alpha = 0.09;
        params.population_size = 100;
        double lowerbound[3] = {-5.12, -5.12, -5.12};
        double upperbound[3] = {5.12, 5.12, 5.12};

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

        return 0;
}
