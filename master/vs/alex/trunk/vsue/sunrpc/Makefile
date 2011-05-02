#CFLAGS=-Wall

all:
	rpcgen messageboard.x
	cc $(CFLAGS) board-client.c messageboard_clnt.c messageboard_xdr.c -o board-client
	cc $(CFLAGS) board-prog.c messageboard_svc.c messageboard_xdr.c -o board-prog

clean:
	rm -rf messageboard_*.c messageboard.h board-client board-prog

.PHONY: all clean

