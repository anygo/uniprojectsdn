#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int strCount = 0;

char** readInput()
{
	char** inputArr;
	int charCount = 0;
	char tmp;

	inputArr = (char**)malloc(sizeof(char*));
	inputArr[strCount] = (char*)malloc(sizeof(char)*101);

	while ((tmp = fgetc(stdin)) != EOF)
	{
		if (tmp == '\n')
		{
			inputArr[strCount][charCount] = '\0';
			/* ueberflussigen speicher frei machen:*/ 
			inputArr[strCount] = realloc(inputArr[strCount], sizeof(char)*(charCount+1));
			charCount = 0;
			strCount++;
			inputArr = realloc(inputArr, sizeof(int)*(strCount+1));
			inputArr[strCount] = (char*)malloc(sizeof(char)*101);
		}
		else
		{
			inputArr[strCount][charCount] = tmp;
			charCount++;
		}
		if (charCount > 100)
		{
			charCount = 0;
			free(inputArr[strCount]);
			inputArr[strCount] = (char*)malloc(sizeof(char)*101);
			while (fgetc(stdin) != '\n')
			{
				/* warten, bis das Wort mit mehr als 100 Zeichen zu Ende ist
				 * und solange nichts tun... */
			}
		}
		
	}
	return inputArr;
}

void printArr(char** arr)
{
	int i = 0;
	int j = 0;
	while (i < strCount)
	{
		if (arr[i][j] != '\0')
		{
			printf("%c", arr[i][j++]);
		}
		else
		{
			printf("\n");
			i++;
			j = 0;
		}
	}
}

static int mystrcmp(const void *p1, const void *p2)
{
	return strcmp(* (char * const *) p1, * (char * const *) p2);
}

int main(int argc, char *argv[])
{	
	int i, j;
	char** inputArr;
	
	if (argc > 1)
	{
		qsort(&argv[1], argc - 1, sizeof(char *), mystrcmp);

		for (j = 1; j < argc; j++)
		{
		puts(argv[j]);
		}
		return 0;
	}

	inputArr = readInput();

	qsort(&inputArr[0], strCount, sizeof(char *), mystrcmp);

	printArr(inputArr);

	for (i = 0; i < strCount + 1; i++)
	{
		free(inputArr[i]);
	}

	free(inputArr);
	return 0;
}
