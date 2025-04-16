#ifndef PRS_H
#define PRS_H

#include <stdint.h>

typedef struct {
        uint32_t dim;
        uint32_t max_iter;
        uint32_t population_size;
} prs_params_t;

/* print shit */

void prs_print_params(const prs_params_t* params);
void prs_print_solution(
        const prs_params_t*     params,
        const double*           solution,
        const double*           score
);

/* prs core stuff*/

double** prs_init_population(
        const prs_params_t*     params, 
        double*                 lowerbound, 
        double*                 upperbound
);

double prs_init_prism_angle(
        const prs_params_t*     params,
        double*                 lowerbound,
        double*                 upperbound
);

void prs_optimizer(
        const prs_params_t*     params,
        double*                 lowerbound,
        double*                 upperbound,
        double*                 best_solution,
        double*                 best_score
);

#endif // PRS_H