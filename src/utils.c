#include <utils.h>

#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

uint8_t gen_random(uint8_t lower, uint8_t upper) {
        if (lower > upper) {
                uint64_t temp = lower;
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

double eval_fitness(const double* x, const uint32_t* dim) {
        double fitness = 0.0;
        for (uint32_t i = 0; i < *dim; i++) {
                fitness += x[i] * x[i];
        }
        return fitness;
}
