CFLAGS=-ansi -g -pedantic -Wall -Werror -D_POSIX_SOURCE -D_XOPEN_SOURCE=500
CC=gcc

.PHONY: all clean install

all: smash-static smash

install: smash smash-static
	mkdir -p ../bin.i386
	mkdir -p ../lib.i386
	cp libsp.so ../lib.i386/
	cp smash ../bin.i386/
	cp libsp.a ../lib.i386/
	cp smash-static ../bin.i386/

smash-static: libsp.a smash.c
	$(CC) -static -L. -o smash-static smash.c -lsp

libsp.a: plist.o halde.o
	ar -r -c -s libsp.a plist.o halde.o

smash: libsp.so smash.c
	$(CC) smash.c -L. -lsp -o smash

libsp.so: plist_so.o halde_so.o
	$(CC) -shared -o libsp.so plist_so.o halde_so.o

plist_so.o: plist.c
	$(CC) -fPIC -c -o plist_so.o plist.c

halde_so.o: halde.c
	$(CC) -fPIC -c -o halde_so.o halde.c

plist.o: plist.c 
	$(CC) $(CFLAGS) -c plist.c

halde.o: halde.c
	$(CC) $(CFLAGS) -c halde.c

clean:
	rm -f plist.o halde.o plist_so.o halde_so.o
	rm -f libsp.a 
	rm -f libsp.so
	rm -f smash-static smash

