CFLAGS=-O3 -pthread
CC=gcc
.PHONY: all clean install

all: wsort

wsort:
	$(CC) $(CFLAGS) -o wsort wsort.c

clean:
	rm -f wsort

install: all
	cp wsort ../bin.i386/wsort
