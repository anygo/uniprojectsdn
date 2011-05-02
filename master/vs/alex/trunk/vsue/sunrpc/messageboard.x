typedef struct message *messagelist;

struct message {
	int uid;
	string title<>;
	string message<>;
	messagelist next;
};

union messagelist_res switch(int retval) {
	case 0: messagelist list;
	default: void;
};

program BOARDPROG {
	version BOARDVERS {
		int post_message(message) = 1;
		messagelist_res get_messages(int) = 2;
	} = 1;
} = 20100514; /* 20000000 - 3fffffff: numbers defined by user */

