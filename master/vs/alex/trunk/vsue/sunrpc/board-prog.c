#include <stdio.h>
#include <string.h>
#include "messageboard.h"

static messagelist msg_list_head = NULL;

int *post_message_1_svc(message *msg, struct svc_req *req)
{
	static int result;
	messagelist node;

	node = (message *)malloc(sizeof(message));
	if(node == NULL) {
		result = -1;
		return &result;
	}
	node->uid = msg->uid;
	node->title = strdup(msg->title);
	node->message = strdup(msg->message);
	if(node->title == NULL || node->message == NULL) {
		result = -1;
		return &result;
	}
	node->next = msg_list_head;
	msg_list_head = node;

	result = 0;	
	return &result;
}

messagelist_res *get_messages_1_svc(int *num_msgs, struct svc_req *req)
{
	int num, i;
	static messagelist_res res;
	messagelist *list, msg, tmp;
	
	num = *num_msgs;
	if(num <= 0) {
		res.retval = -1;
		return &res;
	}

	if(msg_list_head == NULL) {
		res.retval = -2;
		return &res;
	}

	// free previous result
	xdr_free((xdrproc_t)xdr_messagelist_res, (char *)&res);

	tmp = msg_list_head;
	list = &res.messagelist_res_u.list;
	for(i = 0; i < num; i++) {
		msg = *list = (message *)malloc(sizeof(message));
		if(msg == NULL) {
			res.retval = -1;
			return &res;
		}

		msg->uid = tmp->uid;
		msg->title = strdup(tmp->title);
		msg->message = strdup(tmp->message);
		list = &msg->next;

		if(tmp->next != NULL)
			tmp = tmp->next;
		else
			break;
	}
	*list = NULL; 

	res.retval = 0;
	return &res;
}

