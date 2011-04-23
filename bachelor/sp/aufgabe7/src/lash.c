#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <string.h>
#include <sys/types.h>
#include <limits.h>
#include <signal.h>

#include "plist.h"

static void printStat(const char *cmdLine, int status);
static int parseCommandLine(char *buffer, char **argv, char* line);
static void execute(char *commandLine, char **argv, int bg);
static void collectBGProc();
static void prompt();

/*
 * Zerlegt die Kommandozeile an Leerzeichen und Tabs und legt
 * die Parameter in argv ab.
 * 
 * buffer wird dabei zerstoert!
 * Liefert die Anzahl der Parameter zurueck.
 */
static int parseCommandLine(char *buffer, char **argv, char* line) {
	int cnt=0;

	strncpy(buffer, line, ARG_MAX);

	/* buffer ist leer -> kein Argument enthalten */
	argv[0]=NULL;
	if(! *buffer) return 0;

	argv[cnt++] = strtok(buffer, " \t");
	while((argv[cnt] = strtok(NULL, " \t")) != NULL) cnt++;

	/* argv mit NULL Zeiger fuer exec abschliessen */
	argv[cnt] = NULL;

	return cnt;
}

/*
 * Erzeugt einen neuen Prozess, fuehrt das angegebene Programm aus und
 * schreibt den Exitstatus (falls Childprozess tatsaechlich terminierte)
 * auf die Standardausgabe
 */
static void execute(char *commandLine, char **argv, int bg) {
	pid_t pid;
	int status;
	int i;
	int j;
	FILE *fd_in;
	FILE *fd_out;

	switch(pid=fork()) {
		case -1:
			perror("fork");
			exit(EXIT_FAILURE);
		case 0:
			if (bg == 1)
			{ /* nur Hintergrundprozesse sollen SIGINT ignorieren */
			  if (sigignore(SIGINT) == -1)
			  {
				fprintf(stderr, "Fehler bei sigignore()");
			  }
			}
			
			for (i = 0; argv[i] != NULL; i++)
			{
				if (argv[i][0] == '<')
				{
					if (NULL == (fd_in = fopen(&argv[i][1], "r+")))
					{
						perror("fopen");
						exit(EXIT_FAILURE);
					}
					if (-1 == dup2(fileno(fd_in), STDIN_FILENO))
					{
						perror("dup2 stdin");
						exit(EXIT_FAILURE);
					}
					for (j = i; argv[j] != NULL; j++)
					{
						argv[j] = argv[j+1];
					}
					i--;
				}
				else if (argv[i][0] == '>') 
				{
					if (NULL == (fd_out = fopen(&argv[i][1], "w+")))
					{
						perror("fopen");
						exit(EXIT_FAILURE);
					}
					if (-1 == dup2(fileno(fd_out), STDOUT_FILENO))
					{
						perror("dup2 stdout");
						exit(EXIT_FAILURE);
					}
					argv[i] = NULL;
				}
			}
			if (argv[i] == NULL) /* falls danach wirklich nichts mehr kommt */
			{
				execvp(argv[0], argv);
			}
			  perror(argv[0]);
			  exit(EXIT_FAILURE);
	}

	/* don't wait for background process */
	if(bg!=0) {
		if(insertElement(pid, commandLine) < 0) {
			perror("insertElement");
			exit(EXIT_FAILURE);
		}
		return;
	}

	/* do wait for foreground process */
	if (waitpid(pid, &status, 0)==-1) {
		perror("wait for foreground process");
		exit(EXIT_FAILURE);
	}
	printStat(commandLine, status);
}

static void printStat(const char *cmdLine, int status) {
	if (WIFSIGNALED(status)) {
		printf("Signal [%s] = %d",cmdLine,WTERMSIG(status));
		printf("\n");
	} else { 
		printf("Exitstatus [%s] = %d\n",cmdLine,WEXITSTATUS(status));
	}
}

static void collectBGProc() {
	pid_t bgpid;
	int bgstatus;

	while( (bgpid=waitpid(-1, &bgstatus, WNOHANG)) > 0 ) {
		char cmdLine[ARG_MAX];

		if( removeElement(bgpid, cmdLine, ARG_MAX) < 0 ) {
			fprintf(stderr, "cuckold child %d\n", bgpid);
			continue;
		}

		printStat(cmdLine, bgstatus);
	}
}

static void prompt() {
	char  cworkdir[PATH_MAX];
	if (NULL==getcwd(cworkdir, PATH_MAX)) {
		perror("getcwd");
		strcpy(cworkdir,"[unknown]");
	}
	printf("%s: ",cworkdir);
}

void handle_child(int sig)
{
	collectBGProc();
}

void handle_int(int sig)
{
	fprintf(stderr, "Interrupt!\n");
}	

int main() {
	int   arglen;

	char  commandLine[ARG_MAX];
	char  argBuffer[ARG_MAX];
	char* argVector[ARG_MAX/2];
	struct sigaction action;
	struct sigaction actionint;
	int pid;
	int i;
	sigset_t set;

	sigemptyset(&set);
	sigaddset(&set, SIGCHLD);

	if (sigemptyset(&action.sa_mask) == -1)
	{
		fprintf(stderr, "Fehler bei sigemptyset()");
		return EXIT_FAILURE;
	}
	action.sa_flags = (SA_NOCLDSTOP | SA_RESTART);
	action.sa_handler = handle_child;
	if (sigaction(SIGCHLD, &action, NULL) == -1)
	{
		fprintf(stderr, "Fehler bei sigaction");
		return EXIT_FAILURE;
	}

	if (sigemptyset(&actionint.sa_mask) == -1)
	{
		fprintf(stderr, "Fehler bei sigemtyset()");
		return EXIT_FAILURE;
	}
	actionint.sa_flags = SA_RESTART;
	actionint.sa_handler = handle_int;
	if (sigaction(SIGINT, &actionint, NULL) == -1)
	{
		fprintf(stderr, "Fehler bei sigaction");
		return EXIT_FAILURE;
	}


	while(1) {
		collectBGProc();
		prompt();

		/* get command */
		if ( NULL == fgets(commandLine, ARG_MAX, stdin) ) {
			if(feof(stdin)) {
				printf("\n");
				exit(EXIT_SUCCESS);
			} else {
				perror("fgets");
				exit(EXIT_FAILURE);
			}
		}

		/* strip newline */
		if(commandLine[strlen(commandLine)-1] == '\n') {
			commandLine[strlen(commandLine)-1] = '\0';
		}

		if ((arglen=parseCommandLine(argBuffer, argVector, commandLine))<1 ||
				argVector[0]==NULL) continue;

		if (strcmp("cd", argVector[0])==0) {
			if (arglen>2) {
				fprintf(stderr,"cd: too many arguments\n");
				continue;
			}
			if (chdir(argVector[1])) { perror("cd"); }	
		} else if (strcmp("jobs", argVector[0])==0)
		{
			if (-1 == sigprocmask(SIG_BLOCK, &set, NULL))
			{
				perror("sigprocmask SIG_BLOCK");
				continue;
			}
			rewindPList();
			while (nextElement((const char **)argVector, &pid) == 0)
			{
				i = 0;
				printf("[%d] ", pid);
				while (argVector[i] != NULL)
				{
					printf("%s ", argVector[i++]);
				}
				printf("\n");
			}
			if (-1 == sigprocmask(SIG_UNBLOCK, &set, NULL))
			{
				perror("sigprocmask SIG_UNBLOCK");
				exit(EXIT_FAILURE);
			}
		} else {
			int bg=0;
			if(*argVector[arglen-1] == '&') {
				argVector[arglen-1] = NULL;
				if(0==--arglen) continue;
				bg = 1;
			}
			execute(commandLine, argVector, bg);
		}
	}
}

