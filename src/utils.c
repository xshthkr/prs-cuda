#include <utils.h>

#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

uint8_t gen_random_unsigned(uint8_t lower, uint8_t upper) {
        if (lower > upper) {
                uint8_t temp = lower;
                lower = upper;
                upper = temp;
        }

        uint8_t range = upper - lower + 1;
        uint8_t rand_val;
        int fd = open("/dev/urandom", O_RDONLY);
        if (fd < 0) {
                perror("Failed to open /dev/urandom");
                exit(EXIT_FAILURE);
        }

        uint8_t limit = UINT8_MAX - (UINT8_MAX % range);

        do {
                if (read(fd, &rand_val, sizeof(rand_val)) != sizeof(rand_val)) {
                perror("Failed to read random bytes");
                close(fd);
                exit(EXIT_FAILURE);
                }
        } while ((uint8_t)rand_val >= limit);

        close(fd);
        return lower + ((uint8_t)rand_val % range);
}

double gen_random_double(double lower, double upper) {
        uint64_t rand_int;
        int urandom_fd = open("/dev/urandom", O_RDONLY);
        if (urandom_fd < 0) {
                perror("Failed to open /dev/urandom");
                exit(EXIT_FAILURE);
        }

        if (read(urandom_fd, &rand_int, sizeof(rand_int)) != sizeof(rand_int)) {
                perror("Failed to read from /dev/urandom");
                close(urandom_fd);
                exit(EXIT_FAILURE);
        }

        close(urandom_fd);

        // convert to a double in [0, 1]
        double normalized = (rand_int / (double)UINT64_MAX);

        // scale to [lower, upper]
        return lower + normalized * (upper - lower);
}

double eval_fitness(const double* x, const uint32_t dim) {
        double fitness = 0.0;
        for (uint32_t i = 0; i < dim; i++) {
                fitness += x[i] * x[i];
        }
        return fitness;
}

double max(const double* x, const uint32_t dim) {
        double max_val = x[0];
        for (uint32_t i = 1; i < dim; i++) {
                if (x[i] > max_val) {
                        max_val = x[i];
                }
        }
        return max_val;
}

double min(const double* x, const uint32_t dim) {
        double min_val = x[0];
        for (uint32_t i = 1; i < dim; i++) {
                if (x[i] < min_val) {
                        min_val = x[i];
                }
        }
        return min_val;
}
