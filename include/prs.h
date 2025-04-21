#ifndef PRS_H
#define PRS_H

#include <stdint.h>

typedef struct {
        uint32_t                dim;
        uint32_t                max_iter;
        uint32_t                population_size;
        float                   alpha;
} prs_params_t;

/* print stuff */

void prs_print_params(
        const prs_params_t*     params
);

void prs_print_solution(
        const prs_params_t*     params,
        const double*           solution,
        const double*           score
);

/* core prs stuff*/

double** prs_init_incidence_angles(
        const prs_params_t*     params
);

double prs_init_prism_angle();

double** prs_init_emergent_angles(
        const prs_params_t*     params
);

double prs_get_refractive_index(
        double*                 prism_angle,
        double*                 delta,
        const uint32_t          dim
);

double prs_angle_to_solution(
        double*                 angle,
        double*                 lowerbound,
        double*                 upperbound
);

double prs_solution_to_angle(
        double*                 solution,
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