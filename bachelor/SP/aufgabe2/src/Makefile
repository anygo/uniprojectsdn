CFLAGS=-ansi -pedantic -D_POSIX_SOURCE -Wall -Werror
CC=/usr/bin/gcc
.PHONY: all clean install

all: wsort

wsort: wsort.o
	$(CC) $(CFLAGS) -o wsort wsort.o

wsort.o: wsort.c
	$(CC) $(CFLAGS) -c wsort.c

clean:
	rm -f wsort
	rm -f wsort.o

install: all
	cp wsort ../bin.i386/wsort
