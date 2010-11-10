#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include "plist.h"

#define LINE_MAX  sysconf(_SC_LINE_MAX)
#define ARG_MAX  sysconf(_SC_ARG_MAX)


int main()
{
	size_t i;	
	int status;
	pid_t pid;

	char *buf;
	char *path;
	char *buf2;
	char** args;
	
	buf = (char*)malloc(LINE_MAX);
	path = (char*)malloc(LINE_MAX);
	buf2 = (char*)malloc(LINE_MAX);
	args = (char**) malloc(ARG_MAX*sizeof(char*));


	while(1)
	{
		i = 0;
		status = (int)NULL;
		if(0 == getcwd(path,sysconf(_SC_LINE_MAX)))
		{
			perror("getcwd");
			free(buf);
			free(buf2);
			free(path);
			free(args);
			exit(1);
		}
		printf("%s:",path);
		
	 	if(NULL==fgets(buf,sysconf(_SC_LINE_MAX),stdin))
		{
			
			perror("getcwd");
			free(buf);
			free(buf2);
			free(path);
			free(args);
			return 0;
		}
		strcpy(buf2, buf);
		buf2[strlen(buf2)-1] = '\0';

		args[0] = (char*) strtok(buf," ");
		while(args[i] != NULL && i < sysconf(_SC_ARG_MAX))
		{
			i++;
			args[i] = (char*) (strtok(NULL," \t\n"));
		}
		if(strcmp(args[0],"cd") == 0)
		{
			if(args[1][0] == '/')
				chdir((char*) &args[1]);
			else
			{
				strcat(path,"/");
				strcat(path,(char*)args[1]);
				chdir((char*) path);
			}
		}
		else
		{
			pid = fork();
			switch(pid)
			{
			case 0:
				if(-1 == execvp(args[0],&args[0]))
				{
					perror("exec");
					exit(1);
				}
				exit(1);
				break;
			case -1:
				perror("fork");

				perror("getcwd");
				free(buf);
				free(buf2);
				free(path);
				free(args);
				exit(1);
				break;
			default:
				if(strcmp(args[i-1], "&") == 0)
					insertElement(pid,buf2);			
				else
				{
					waitpid(pid,&status,0);
					if(WIFEXITED(status))
						printf("Exitstatus [%s] = %d\n",buf2,WEXITSTATUS(status));
					else if(WIFSIGNALED(status))
						printf("Signal [%s] = %d\n",buf2,WTERMSIG(status));
				}
			}
		}
		do
		{
			status = (int)NULL;
			pid=waitpid(-1,&status,WNOHANG);
			if(removeElement(pid,buf,sysconf(_SC_LINE_MAX)) < 0) break;
			if(WIFEXITED(status))
				printf("Exitstatus [%s] = %d\n",buf,WEXITSTATUS(status));
			else if(WIFSIGNALED(status))
				printf("Signal [%s] = %d\n",buf,WTERMSIG(status));
		}while(pid != 0 && pid != -1);
	}

	perror("getcwd");
	free(buf);
	free(buf2);
	free(path);
	free(args);
	return 0;
}
