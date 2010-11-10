#include <stdio.h>
#include <stdlib.h>
#include "plist.h"
#include <string.h>

/* struct listelement */
struct listelement
{
    pid_t pid;
	char *cmdline;
    struct listelement *next;
};

/* type definitions */
typedef struct listelement listelement;

/* static variables */
static listelement *head = NULL;

/* help function */
/* checks if element with pid pid exists in the list
 * returns element with pid pid on success, else NULL */
listelement * elementExists(pid_t pid)
{
	listelement *tmp = head;
	
	while (tmp != NULL)
	{
		if (tmp->pid == pid)
			return tmp;
		tmp = tmp->next;
	}
	return NULL;
}

/* slist-functions */
int insertElement(pid_t pid, const char *commandLine)
{   
    listelement *newElement; 	
	int cmdLineLength = strlen(commandLine) + 1; /* wegen '\0' */
	char* cmdline; 


	newElement = (listelement *)malloc(sizeof(listelement));
	if (newElement == NULL)
		return -2;
	
	cmdline = (char *)malloc(sizeof(char)*cmdLineLength);
	if (cmdline == NULL)
	{
		free(newElement);
		return -2;
	}
	strcpy(cmdline, commandLine);

	newElement->pid = pid;
	newElement->cmdline = cmdline;
	newElement->next = NULL;
	

	if (elementExists(pid) != NULL)
	{
		free(cmdline);
		free(newElement);
		return -1;
	}

    if (pid < 0)
    {   
		free(cmdline);
        free(newElement);
		fprintf(stderr, "err (insertElement): only pid's > 0!\n");
        return -1; 
    }
	
	newElement->next = head;
	head = newElement;
	return pid;
}

int removeElement(pid_t pid, char *commandLineBuffer, size_t bufferSize)
{
    listelement *tmp = head;
	listelement *toRemove;
	int cmdLineLength;

	if (head == NULL)
	{
		return -1;
	}

	toRemove = elementExists(pid);

		if (toRemove == NULL)
	{
		return -1;
	}
	
	if (commandLineBuffer == NULL)
	{
		return -100; /* negativer Wert bei error */
	}

    if (pid < 0)
    {
        printf("err (removeElement): only pids > 0!\n");
    }

	if (tmp != toRemove)
	{	
		while (tmp->next != toRemove)
		{
			tmp = tmp->next;
		}
	}
	
	tmp->next = toRemove->next;

	cmdLineLength = (strlen(toRemove->cmdline)+1);

	if (cmdLineLength <= bufferSize)
	{
		strcpy(commandLineBuffer, toRemove->cmdline);
	}
	else
	{
		strncpy(commandLineBuffer, toRemove->cmdline, bufferSize-1);
		commandLineBuffer[bufferSize-1] = '\0';
	}
	
	if (head == toRemove)
	{	
		head = head->next;;
	}
	free(toRemove->cmdline);
	free(toRemove);
	

	if (cmdLineLength <= bufferSize);
		return cmdLineLength;	
	return bufferSize;
}

void printList()
{
    listelement *tmp = head;
    while (tmp != NULL)
    {
        printf("pid %d (%s); ", tmp->pid, tmp->cmdline);
        tmp = tmp->next;
    }
    printf("\n");
}

int main()
{
	char *a = "hallo";
	char *b = "dallo";
	char *c = "deppo";
	char *ar = (char *)malloc(sizeof(char)*10);
	char *br = (char *)malloc(sizeof(char)*10);
	char *cr = (char *)malloc(sizeof(char)*10);
	printList();
	insertElement(1000, a);
	printList();
	insertElement(1001, b);
	printList();
	insertElement(1002, c);
	printList();
	printf("rem %i (%s)\n", removeElement(1002, br, 10), br);
	printList();
	printf("rem %i (%s)\n", removeElement(1000, ar, 10), ar);
	printList();
	printf("rem %i (%s)\n", removeElement(1001, cr, 10), cr);
	printList();
	insertElement(1, a);
	printList();
	insertElement(1003,a);
	insertElement(1004,a);
	insertElement(1005,a);
	printList();
	removeElement(1000,ar,10);
	removeElement(1003,ar,10);
	removeElement(1004,ar,10);
	removeElement(1005,ar,10);
	removeElement(1,ar,10);
	removeElement(1001,ar,10);
	removeElement(1002,ar,10);
printList();



return 0;
}
