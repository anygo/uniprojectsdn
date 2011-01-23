.PHONY: all clean
.SUFFIXES: .c .o

CC = gcc 

CFLAGS = -ansi -g -pedantic -Wall -Werror -D_POSIX_SOURCE -D_XOPEN_SOURCE=500

all: myfind

myfind: myfind.o
	$(CC) $(CFLAGS) -o myfind myfind.o

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f myfind myfind.o

# dependencies
myfind.o:   myfind.c

