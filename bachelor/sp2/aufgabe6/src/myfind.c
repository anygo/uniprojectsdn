#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <fnmatch.h>
#include <string.h>
#include <limits.h>
#include <bits/local_lim.h>

static int type_test(char type, char d_type);
static int pattern_test(char* pattern, char* d_name);
static int link_test(int links, nlink_t st_nlink);
static int suche(unsigned int depth, unsigned int maxdepth, char* path, char* pattern, char type, int links);
static char *getPath(char *path); 

int main(int argc, char* argv[])
{
	unsigned int tiefe;
	char* pattern;
	int pathcount;
	char *path;
	char type;
	int links;
	int i;
	int allpathsread;
	int error;
	struct stat stat_buf;

	error = 0;
	path = (char *)malloc(LINE_MAX);
	path[0] = '.';
	path[1] = '/';
	path[2] = '\0';

	pathcount = 0;
	pattern = NULL;
	tiefe = 0xffffffff;
	links = -1;
	type = 'a';
	allpathsread = 0;

	for(i=1; i<argc; i++)
	{
		if(argv[i][0] != '-')
		{
			pathcount++;
			if (allpathsread == 1)
			{
				printf("myfind path.. [-maxdepth n] [-name pattern] [-type {d,f}] [-links n]\n");
				return EXIT_FAILURE;
			}
		}
		else if(strcmp(argv[i],"-maxdepth") == 0)
		{
			if (i + 1 >= argc) {error = 1; break;}
			tiefe = atoi(argv[++i]);
			allpathsread = 1;
		}
		else if(strcmp(argv[i],"-name") == 0)
		{
			if (i + 1 >= argc) {error = 1; break;}
			pattern = (char*) malloc(strlen(argv[i+1]) +1);
			strcpy(pattern,argv[++i]);
			allpathsread = 1;
		}
		else if(strcmp(argv[i],"-type") == 0)
		{
			if (i + 1 >= argc) {error = 1; break;}
			type = argv[++i][0];
			if(type != 'f' && type != 'd')
			{
				error = 1;
				break;
			}
			allpathsread = 1;
		}
		else if(strcmp(argv[i],"-links") == 0)
		{
			if (i + 1 >= argc) {error = 1; break;}
			links = atoi(argv[++i]);
			allpathsread = 1;
		}
		else
		{
			printf("myfind path.. [-maxdepth n] [-name pattern] [-type {d,f}] [-links n]\n");
			return EXIT_FAILURE;
		}
	
	}
	
	if (error == 1)
	{	
		fprintf(stderr, "myfind: Fehler bei Argumenten\n");
		return EXIT_FAILURE;		
	}

	if (tiefe == 0)
	{
		for (i = 0; i < pathcount; i++)
		{
			strcpy(path, argv[i+1]);
			if ((lstat(path,&stat_buf)) == -1) /*wozu ???*/
			{
				perror("lstat");
				continue;
			}
			if(pattern_test(pattern,path))
				printf("%s\n",path);
		}
		return EXIT_SUCCESS;
	}

	for (i = 0; i < pathcount; i++)
	{
		strcpy(path, argv[i+1]);
		suche(0,tiefe-1,path,pattern,type,links);
	}
	if (pathcount == 0)
	{
		suche(0,tiefe-1,path,pattern,type,links);
	}
	getPath(path);
	free(path);
	free(pattern);
	return EXIT_SUCCESS;
}

static int suche(unsigned int depth, unsigned int maxdepth, char* pfad, char* pattern, char type, int links)
{
	DIR* dir;
	struct dirent* dent;
	char* pfad_neu;
	struct stat stat_buf;

	if(depth > maxdepth)
		return 0;
	
	/*if (depth >= OPEN_MAX)
	{
		fprintf(stderr, "maxdepth zu hoch gewaehlt");
		exit(EXIT_FAILURE);
	} */

	/* Wieso findet der Compiler OPEN_MAX nicht? sollte in limits.h sein? */

	depth++;
	dir = opendir(pfad);
	if(dir == NULL)
	{
		if(errno == ENOTDIR)
		{
			lstat(pfad,&stat_buf);
			if(link_test(links, stat_buf.st_nlink) && pattern_test(pattern, pfad) && type != 'd')
				printf("%s\n",pfad);
			closedir(dir);
			return 0;

		}
		else {
		fprintf(stderr, "find: %s: %s\n",pfad,strerror(errno));
		closedir(dir);
		return -1;
		}
	}
	errno = 0;
	while((dent = readdir(dir)) != NULL)
	{
		if(errno != 0)
		{
			perror("readdir");
			closedir(dir);
			errno = 0;
			closedir(dir);
			return -1;
		}
		if ((pfad_neu = (char *)malloc(LINE_MAX)) == NULL)
		{
			perror("malloc");
			closedir(dir);
			exit(1);
		}
		strcpy(pfad_neu,pfad);
		getPath(pfad_neu);
		strcat(pfad_neu,dent->d_name);
		if ((lstat(pfad_neu,&stat_buf)) == -1)
		{
			perror("lstat");
			free(pfad_neu);
			errno = 0;
			continue;
		}
		if(depth == 1 && strcmp(dent->d_name, "." ) == 0 &&
			pattern_test(pattern,pfad) && 
			type_test(type,dent->d_type) && 
			link_test(links, stat_buf.st_nlink))
				printf("%s\n",pfad);
		if (strcmp(dent->d_name, ".") == 0 || strcmp(dent->d_name, "..") == 0)
		{
			errno = 0;
			free(pfad_neu);
			continue;
		}
		if(pattern_test(pattern,dent->d_name) && type_test(type,dent->d_type)
				&& link_test(links, stat_buf.st_nlink))
			printf("%s\n",pfad_neu);

		/*if (!(S_ISDIR(stat_buf.st_mode))) continue;
		if (!(S_ISREG(stat_buf.st_mode))) continue;*/
		if(dent->d_type == 4 && maxdepth > 0)	/*4 => Verzeichnis*/
			suche(depth,maxdepth,pfad_neu,pattern,type,links);
		free(pfad_neu);
		errno = 0;
	}
	if(closedir(dir) != 0)
	{
		perror("closedir");
		return -1;
	}
	return 0;
}

static int pattern_test(char* pattern, char* d_name)
{
	if(pattern == NULL)
		return 1;
	else if(fnmatch(pattern,d_name,FNM_PATHNAME) == 0)
		return 1;
	return 0;
}

static int type_test(char type, char d_type)
{
	if(type == 'a')
		return 1;
	else if(type == 'd' && d_type == 4)
		return 1;
	else if(type == 'f' && d_type == 8)
		return 1;
	return 0;
}

static int link_test(int links, nlink_t st_nlink)
{
	if(links == -1)
		return 1;
	else if(links == st_nlink)
		return 1;

	return 0;
}

static char *getPath(char *path)
{
    int len;

    len = (int)strlen(path);

    if (path[len - 1] == '/')
    { 
        return path;
    }   
    else
    { 
        path[len] = '/';
        path[len + 1] = '\0';
    }   
    return path;
}

