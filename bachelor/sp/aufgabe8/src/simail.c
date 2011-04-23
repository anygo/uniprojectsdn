#include <stdio.h>
#include <stdlib.h>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>
#include <netdb.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>

#define PRINTUSAGE fprintf(stderr, "Usage: ./simail [-s subject] <address>\n");
#define dbg 0

extern int h_errno;


static void handle_server_message(char *messagebuf, int s_fd, const char *correctstatus)
{
	char smallbuf;
	int i;

	smallbuf = '\0';
	i = 0;
	while(smallbuf != '\n')
	{
		if (read(s_fd, (void *)&smallbuf, 1) == -1)
		{
			perror("read()");
			close(s_fd);
			exit(EXIT_FAILURE);
		}
		messagebuf[i] = smallbuf;
		errno = 0;
		if (dbg)	putc(smallbuf, stdout);
		i++;
	}
	messagebuf[i] = '\0';
	if (strncmp(messagebuf, correctstatus, 3) != 0)
	{
		fprintf(stderr, "%sError. Exiting.\n", messagebuf);
		exit(EXIT_FAILURE);
	}
}

static void mywrite(int fd, const void *buf, size_t count)
{
	if (write(fd, buf, count) == -1)
	{
		perror("write()");
		exit(EXIT_FAILURE);
	}
}

int main(int argc, const char* argv[])
{
	char hostname[LINE_MAX];
	struct hostent *hostent;
	struct hostent *hostent_server;
	char fqdn[LINE_MAX];
	char mailfrom[LINE_MAX];
	char fullname[LINE_MAX];
	struct passwd *mypwd;
	const char *subject;
	const char *to;
	struct sockaddr_in sin;
	int s_fd;
	struct sockaddr_in server;
	char messagebuf[LINE_MAX];
	char tmp;

	/* initialize structs */
	memset(&sin, 0, sizeof(sin));
	memset(&server, 0, sizeof(server));
	memset(&hostname, 0, LINE_MAX);
	memset(&fqdn, 0, LINE_MAX);
	memset(&mailfrom, 0, LINE_MAX);
	memset(&fullname, 0, LINE_MAX);
	memset(&messagebuf, 0, LINE_MAX);
	
	/* check cmdline parameters */
	subject = NULL;
	if (argc == 4)
	{
		if (strcmp(argv[1], "-s") == 0)
		{
			subject = argv[2];
			to = argv[3];
		}
		else 
		{
			PRINTUSAGE
			return EXIT_FAILURE;
		}
	}
	else if (argc == 2)
	{
		to = argv[1];
	}
	else
	{
		PRINTUSAGE
		return EXIT_FAILURE;
	}

	/* get FQDN */
	if (gethostname(hostname, LINE_MAX) != 0)
	{
		perror("gethostname()");
		return EXIT_FAILURE;
	}
	if ((hostent = gethostbyname(hostname)) == NULL)
	{
		herror("gethostbyname()");
		return EXIT_FAILURE;
	}
	strcpy(fqdn, hostent->h_name);
	
	/* get fullname and string for "MAIL FROM" line */
	if ((mypwd = getpwuid(getuid())) == NULL)
	{
		perror("getpwuid");
		return EXIT_FAILURE;
	}
	strcpy(fullname, mypwd->pw_gecos);
	if (strtok(fullname, ",") == NULL)
	{
		fprintf(stderr, "Err: strtok()\n");
		return EXIT_FAILURE;
	}
	strcpy(mailfrom, "<");
	strcat(mailfrom, mypwd->pw_name);
	strcat(mailfrom, "@");
	strcat(mailfrom, fqdn);
	strcat(mailfrom, ">");

	strcat(fullname, " ");
	strcat(fullname, mailfrom);
	
	/* prepare communication */
	if ((s_fd = socket(PF_INET, SOCK_STREAM, 0)) == -1)
	{
		perror("socket()");
		return EXIT_FAILURE;
	}
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = htonl(INADDR_ANY);
	sin.sin_port = htons(0);
	if (bind(s_fd, (struct sockaddr *) &sin, sizeof(sin)) != 0)
	{
		perror("bind()");
		return EXIT_FAILURE;
	}
	
	if ((hostent_server = gethostbyname("faui03.informatik.uni-erlangen.de")) == NULL)
	{
		herror("gethostbyname()");
		return EXIT_FAILURE;
	}
	memcpy(&server.sin_addr, hostent_server->h_addr, hostent_server->h_length); 
	server.sin_family = AF_INET; 
	server.sin_port = htons(25);
	server.sin_family = AF_INET;
	
	/* connect to server */
	if (connect(s_fd, (struct sockaddr *) &server, sizeof(server)) != 0)
	{
		perror("connect()");
		return EXIT_FAILURE;
	}
	
	memset(&messagebuf, 0, LINE_MAX);

	
	handle_server_message(messagebuf, s_fd, "220");
	
	/* send HELO */
	mywrite(s_fd, "HELO ", 5);
	mywrite(s_fd, fqdn, strlen(fqdn));
	mywrite(s_fd, "\n", 1);

	if (dbg)	mywrite(STDOUT_FILENO, "HELO ", 5);
	if (dbg)	mywrite(STDOUT_FILENO, fqdn, strlen(fqdn));
	if (dbg)	mywrite(STDOUT_FILENO, "\n", 1);

	
	handle_server_message(messagebuf, s_fd, "250");

	/* send MAIL FROM */
	mywrite(s_fd, "MAIL FROM: ", 11);
	mywrite(s_fd, mailfrom, strlen(mailfrom));
	mywrite(s_fd, "\n", 1);

	if (dbg)	mywrite(STDOUT_FILENO, "MAIL FROM: ", 11);
	if (dbg)	mywrite(STDOUT_FILENO, mailfrom, strlen(mailfrom));
	if (dbg)	mywrite(STDOUT_FILENO, "\n", 1);

	
	handle_server_message(messagebuf, s_fd, "250");
	
	/* send RCPT TO */
	mywrite(s_fd, "RCPT TO: ", 9);
	mywrite(s_fd, to, strlen(to));
	mywrite(s_fd, "\n", 1);

	if (dbg)	mywrite(STDOUT_FILENO, "RCPT TO: ", 9);
	if (dbg)	mywrite(STDOUT_FILENO, to, strlen(to));
	if (dbg)	mywrite(STDOUT_FILENO, "\n", 1);

	
	handle_server_message(messagebuf, s_fd, "250");

	/* send DATA */
	mywrite(s_fd, "DATA\n", 5);

	if (dbg)	mywrite(STDOUT_FILENO, "DATA\n", 5);


	handle_server_message(messagebuf, s_fd, "354");

	/* send From */
	mywrite(s_fd, "From: ", 6);
	mywrite(s_fd, fullname, strlen(fullname));
	mywrite(s_fd, "\r\n", 2);

	if (dbg)	mywrite(STDOUT_FILENO, "From: ", 6);
	if (dbg)	mywrite(STDOUT_FILENO, fullname, strlen(fullname));
	if (dbg)	mywrite(STDOUT_FILENO, "\r\n", 2);

	/* send To */
	mywrite(s_fd, "To: <", 5);
	mywrite(s_fd, to, strlen(to));
	mywrite(s_fd, ">\r\n", 3);
	
	if (dbg)	mywrite(STDOUT_FILENO, "To: <", 5);
	if (dbg)	mywrite(STDOUT_FILENO, to, strlen(to));
	if (dbg)	mywrite(STDOUT_FILENO, ">\r\n", 3);

	/* send Subject */
	if (subject != NULL)
	{
		mywrite(s_fd, "Subject: ", 9);
		mywrite(s_fd, subject, strlen(subject));
		mywrite(s_fd, "\r\n", 2);
	
	if (dbg)	mywrite(STDOUT_FILENO, "Subject: ", 9);
	if (dbg)	mywrite(STDOUT_FILENO, subject, strlen(subject));
	if (dbg)	mywrite(STDOUT_FILENO, "\r\n", 2);
	}

	/* send seperator-line */
	mywrite(s_fd, "\r\n", 2);

	if (dbg)	mywrite(STDOUT_FILENO, "\r\n", 2);

	/* read and send Body */
	while (errno = 0, (tmp = fgetc(stdin)) != EOF)
	{
		if (errno != 0)
		{
			perror("fgetc()");
			return EXIT_FAILURE;
		}
		mywrite(s_fd, &tmp, 1);
	if (dbg)	mywrite(STDOUT_FILENO, &tmp, 1);
	}

	/* send .-Line */
	mywrite(s_fd, "\n.\r\n", 4);

	if (dbg)	mywrite(STDOUT_FILENO, "\n.\r\n", 4);

	handle_server_message(messagebuf, s_fd, "250");

	/* send QUIT */
	mywrite(s_fd, "QUIT\r\n", 6);
	
	if (dbg)	mywrite(STDOUT_FILENO, "QUIT\r\n", 6);

	handle_server_message(messagebuf, s_fd, "221");

	close(s_fd);
	return EXIT_SUCCESS;
}
