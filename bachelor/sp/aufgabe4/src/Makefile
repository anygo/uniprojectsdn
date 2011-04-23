.PHONY: all clean
.SUFFIXES: .c .o

CC = gcc

#
# sbrk(2) is not in POSIX. We need _XOPEN_SOURCE=500.
#
CFLAGS = -ansi -g -pedantic -Wall -Werror -D_XOPEN_SOURCE=500
#CFLAGS = -ansi -O2 -pedantic -Wall -Werror -D_XOPEN_SOURCE=500

all: bug malloc-test

bug: bug.c halde.h halde.o
	$(CC) $(CFLAGS) -o $@ bug.c halde.o

malloc-test: malloc-test.o halde.o
	$(CC) $(CFLAGS) -o $@ malloc-test.o halde.o

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f bug halde.o bug.o malloc-test malloc-test.o

# dependencies
halde.o:       halde.h halde.c
bug.o:         halde.h bug.c
malloc-test.o: malloc-test.c

