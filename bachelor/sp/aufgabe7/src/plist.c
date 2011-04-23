#include <stdlib.h>
#include <sys/types.h>
#include <string.h>

#include "plist.h"

static struct qel {
	pid_t pid;
	char *cmdLine;
	struct qel *next;
} *head, *pos;

int insertElement(pid_t pid, const char *cmdLine) {
	struct qel **lastp = &head;
	struct qel *new;

	while( (*lastp != NULL) && ((*lastp)->pid < pid) ) {
		lastp = &((*lastp)->next);
	}

	/* PID existiert bereits */
	if ( *lastp != NULL && (*lastp)->pid == pid ) {
		return -1;
	}
	
	new = malloc(sizeof(struct qel));
	if ( NULL == new ) { return -2; }

	new->cmdLine = malloc(strlen(cmdLine)+1);
	if( NULL == new->cmdLine ) {
		free(new);
		return -2;
	}
	strcpy(new->cmdLine, cmdLine);
	new->pid = pid;

	/* einhaengen des neuen Elements */
	new->next = *lastp;
	*lastp = new;

	return pid;
}

int removeElement(pid_t pid, char *buf, size_t buflen) {
	struct qel **lastp = &head;
	struct qel *del;
	int cll;

	while((*lastp != NULL) && ((*lastp)->pid < pid) ) {
		lastp = &((*lastp)->next);
	}

	/* PID not found */ 
	if( *lastp==NULL || (*lastp)->pid != pid ) { 
		return -1;
	}

	/* aushaengen */
	del = *lastp;
	*lastp = del->next;

	strncpy(buf, del->cmdLine, buflen);
	if( buflen>0 ) {
		buf[buflen-1]='\0';
	}
	cll = strlen(del->cmdLine);

	/* Speicher freigeben */
	free(del->cmdLine);
	free(del);

	/* Echte Laenge zurueckliefern */
	return cll;
}

void rewindPList() {
	pos = head;
}

int nextElement(const char **cmdLine, pid_t *pid) {
	if( NULL == pos ) {
		return -1;
	}

	*cmdLine = pos->cmdLine;
	*pid = pos->pid;
	pos = pos->next;
	return 0;
}



