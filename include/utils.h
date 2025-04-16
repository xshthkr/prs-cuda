#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

uint8_t gen_random_unsigned(
        uint8_t lowerbound,
        uint8_t upperbound
);

double gen_random_double(
        double lowerbound,
        double upperbound
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