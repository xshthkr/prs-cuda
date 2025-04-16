#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

int8_t gen_random(
        int8_t lowerbound,
        int8_t upperbound
);

double eval_fitness(
        const double* solution,
        const uint32_t dim
);

double max(
        const double* x,
        const uint32_t dim
);

double min(
        const double* x,
        const uint32_t dim
);

#endif // UTILS_H