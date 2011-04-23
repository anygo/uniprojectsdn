CFLAGS=-Wall -Werror -Wextra -ansi -pedantic -O3 -D_XOPEN_SOURCE=500
CC=gcc

.PHONY: all clean install doc

all: lash 

clean:
	rm -f lash 

install: all
	strip --strip-unneeded lash
	mkdir -p /proj/i4sp/pub/aufgabe7
	chgrp i4sp /proj/i4sp/pub/aufgabe7
	chmod 2775 /proj/i4sp/pub/aufgabe7
	cp Makefile plist.[ch] lash /proj/i4sp/pub/aufgabe7
	chmod 755 /proj/i4sp/pub/aufgabe7/lash
	chmod 664 /proj/i4sp/pub/aufgabe7/plist.[ch] /proj/i4sp/pub/aufgabe7/Makefile

lash: lash.o plist.o
	$(CC) $(CFLAGS) -o $@ $^

doc:
	doxygen doxy.conf

