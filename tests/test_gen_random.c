#include <utils.h>

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

int main () {

        /* test 1 */
        uint8_t lower = 0;
        uint8_t upper = 90;
        uint8_t random_value = gen_random(lower, upper);
        assert(random_value >= lower && random_value <= upper);

        // int loop = 0;
        // while (loop < 100) {
        //         random_value = gen_random(lower, upper);
        //         printf("%d\n", random_value);
        //         fflush(stdout);
        //         assert(random_value >= lower && random_value <= upper);
        //         loop++;
        // }

        printf("All tests passed.\n");
        return 0;
}