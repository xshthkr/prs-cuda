CC = gcc
CFLAGS = -Wall -Wpedantic -Wextra -Werror -std=c99 -g

clean:
	rm -f src/*.o bin/*