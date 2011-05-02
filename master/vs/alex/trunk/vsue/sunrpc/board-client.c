#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "messageboard.h"

int main(int argc, char **argv)
{
	int op = -1;
	char *server;
	int uid;
	char *title;
	char *message;
	int num_msgs = -1;
	CLIENT *clnt;

	/* post message */
	if((argc == 6) && (!strcmp(argv[2], "POST"))) {
		op = 1; 
		server = argv[1];
		uid = atoi(argv[3]);
		if(uid <= 0) {
			fprintf(stderr, "Error: invalid user-id <uid>\n");
			exit(EXIT_FAILURE);
		}
		title = argv[4];
		message = argv[5];

	/* get message */
	} else if((argc == 4) && (!strcmp(argv[2], "GET"))) {
		op = 2;
		server = argv[1];
		num_msgs = atoi(argv[3]);
		if(num_msgs <= 0) {
			fprintf(stderr, "Error: invalid number <n> of messages\n\n");
			exit(EXIT_FAILURE);
		}

	/* print usage */
	} else {
		fprintf(stderr, "usage:\n");
		fprintf(stderr, "\t%s <servername> POST <uid> <title> <message>\n", argv[0]);
		fprintf(stderr, "\t%s <servername> GET <n>\n", argv[0]);
		exit(EXIT_FAILURE);

	}

	/* create client handle used for calling RPs on the server */
	clnt = clnt_create(server, BOARDPROG, BOARDVERS, "tcp");
	if(clnt == NULL) {
		/* Couldn't establish connection with server */
		clnt_pcreateerror(server);
		exit(EXIT_FAILURE);
	}

	/* post message */
	if(op == 1) {
		int *result;
		struct message msg_to_send = {uid, title, message, NULL};

		/* Call the remote procedure "post_message" on the server */
		result = post_message_1(&msg_to_send, clnt);
		if(result == NULL) {
			/* An error occurred while calling the server */
			clnt_perror(clnt, server);
			exit(EXIT_FAILURE);
		}
		if(*result != 0) {
			/* result == 0 means no error occured */
			fprintf(stderr, "Received wrong result from server\n");
			exit(EXIT_FAILURE);
		}
		printf("OK\n");
	}

	/* get and print messages */
	if(op == 2) {
		messagelist_res *result;
		messagelist msg;

		/* Call the remote procedure "get_message" on the server */
		result = get_messages_1(&num_msgs, clnt);
		if(result == NULL) {
			/* An error occurred while calling the server */
			clnt_perror(clnt, server);
			exit(EXIT_FAILURE);
		}

		/* result->retval == -2 indicates an empty board */
		if(result->retval == -2) {
			fprintf(stderr, "No messages found on the server\n");
			exit(EXIT_SUCCESS);
		}

		if(result->retval == -1) {
			fprintf(stderr, "Received wrong result from server\n");
			exit(EXIT_FAILURE);
		}

		for(msg = result->messagelist_res_u.list; msg != NULL; msg = msg->next) {
			printf("%d %s %s\n", msg->uid, msg->title, msg->message);
		}

		xdr_free((xdrproc_t)xdr_messagelist_res, (char *)&result);	
	}

	clnt_destroy(clnt);
	exit(EXIT_SUCCESS);
}

