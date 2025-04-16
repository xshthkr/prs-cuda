#include <utils.h>

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

int main () {

        /* test 1 */
        uint8_t lower = 0;
        uint8_t upper = 90;
        uint8_t random_value = gen_random_unsigned(lower, upper);
        assert(random_value >= lower && random_value <= upper);

        // int loop = 0;
        // while (loop < 100) {
        //         random_value = gen_random(lower, upper);
        //         printf("%d\n", random_value);
        //         fflush(stdout);
        //         assert(random_value >= lower && random_value <= upper);
        //         loop++;
        // }

        /* test 2 */
        double lower_d = -1.0;
        double upper_d = 1.0;
        double random_value_double = gen_random_double(lower_d, upper_d);
        assert(random_value_double >= lower_d && random_value_double <= upper_d);

        printf("All tests passed.\n");
        return 0;
}