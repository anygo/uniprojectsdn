#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <fcntl.h>
#include <sys/stat.h>				

#define CRYPT 1
#define DECRYPT 0

static int xor(unsigned char a, unsigned char b)
{
	return (~a & b) | (a & ~b);
}

static void swap_char_p(char *a, char *b)
{
	char *tmp = a;
	a = b;
	b = tmp;
}

static void crypt_byte(unsigned char src, unsigned char key, unsigned char *crypted)
{
	crypted[0] = xor(src, key);
}

static void pic_crypt_byte(unsigned char src, unsigned char key, unsigned char *crypted)
{
	if (src == ' ' && key == ' ') crypted[0] = '#';
	else if (src == ' ' && key == '#') crypted[0] = ' ';
	else if (src == '#' && key == ' ') crypted[0] = ' ';
	else if (src == '#' && key == '#') crypted[0] = '#';
	else crypted[0] = '\n';
}

static void decrypt_byte(unsigned char crypted, unsigned char key, unsigned char *decrypted)
{
	decrypted[0] = xor(crypted, key);
}

static void pic_decrypt_byte(unsigned char crypted, unsigned char key, unsigned char *decrypted)
{
	if (crypted == ' ' && key == ' ') decrypted[0] = '#';
	else if (crypted == ' ' && key == '#') decrypted[0] = ' ';
	else if (crypted == '#' && key == ' ') decrypted[0] = ' ';
	else if (crypted == '#' && key == '#') decrypted[0] = '#';
	else decrypted[0] = '\n';
}

static void keygen(int keylength, char *keyfile)
{
	int i;
	char tmp;
	int fd;
	FILE *keyfile_p;
	FILE *random_p;

	random_p = fopen("/dev/random", "r");
	
	keyfile_p = fopen(keyfile, "w+");
	
	i = 0;
	fflush(random_p);
	while (i < keylength)
	{
		tmp = fgetc(random_p);
		fputc(tmp, keyfile_p);
		i++;
		printf("%i/%i Byte written\r", i, keylength);
	}

	/*srand((unsigned) time(NULL)); 
	for (i = 0; i < keylength; i++)
	{
		tmp = (char) (rand() % (int)pow(2, 8));
		fputc(tmp, keyfile_p);
	}*/
	fclose(keyfile_p);
	fclose(random_p);
}

static void crypt_file(char *inputfile, char *keyfile, char *outputfile, int mode)
{
	char outbuf[1];
	char tmp;
	char picbuf;
	int i;
	int j;
	int c;
	int l;
	int m;
	int magic = 3;
	struct stat info;
	off_t inputsize;
	off_t outputsize;
	off_t keysize;
	char *key;
	FILE *inputfile_p;
	FILE *keyfile_p;
	FILE *outputfile_p;
	FILE *pic;

	if (stat(inputfile, &info) != 0)
	{
		perror(inputfile);
		exit(EXIT_FAILURE);
	}
	inputsize = outputsize = info.st_size;
	inputfile_p = fopen(inputfile, "r");
	outputfile_p = fopen(outputfile, "w+");

	if (stat(keyfile, &info) != 0)
	{
		perror(keyfile);
		exit(EXIT_FAILURE);
	}
	keysize = info.st_size;
	keyfile_p = fopen(keyfile, "r");
	key = (char *) malloc(keysize);
	fread(key, 1, keysize, keyfile_p);
	fclose(keyfile_p);

	if (magic == 3)
	{
		pic = fopen("pic2", "r");
	}

	i = 0;
	j = (int)pow((unsigned int)key[0], (keysize % 3) + 1) % keysize;
	c = 0;
	l = 0;
	printf("\n");
	fflush(stdout);
	while (i < inputsize)
	{
		if (j >= keysize)
		{
			j = 0;
		}
		fscanf(inputfile_p, "%c", &tmp);
		crypt_byte(tmp, key[j], outbuf);
		fprintf(outputfile_p, "%c", outbuf[0]);

		if (key[j] % 2 == 0)
		{
			key[j] += (i % 128) - j;
			swap_char_p(&key[j], &key[(int)pow((double)i, 0.5) % keysize]);
		}
		else
		{
			key[j] = ~key[j] + i;
			swap_char_p(&key[j], &key[(int)pow((double)j, (double)(i % 4)+1) % keysize]);
		}

		/* Schluessel anzeigen lassen (nur fuer sehr kurze Schluessel, sonst viel zu viel
		 * Kosten, Progress Bar muss auskommentiert sein!! */
		/*for (m = 0; m < keysize; m++)
		{
			printf("%i ", (int) key[m]);
		}
		printf("\n");*/
		


		/* Gangsterhackige Progress Bar - deluxe v. 01.000 */
		if (i % (1024*10) == 0)
		{
			printf("Completed: %d/%d KB\r", i/1024, inputsize/1024);
			fflush(stdout);
			}
		if(magic == 1)
		{
			if((i % (inputsize/100) == 0))
			{
				if(c < 99)
				{
					printf("Completed: %d %c			[%d/%d] KB\r",++c,37,i/1024,inputsize/1024);
					fflush(stdout);
				} else if(c == 99)
				{
					c = 100;
					printf("\rCompleted: %d %c		   [%d/%d] KB      \n",100,37,inputsize/1024,inputsize/1024);
					fflush(stdout);
				}
			}
		} 
		if(magic == 2)
		{
			if((i % (inputsize/20) == 0))
			{
				printf("[\033[;1;32m");
				for(l = 0;l <= c;l++)
				{
					printf("#");
				}
				for (l; l <= 20; l++)
				{
					printf(" ");
				}
				printf("\033[0m]       [%d/%d] KB written\r",(c++)*(inputsize/(1024*20)),inputsize/1024);
				fflush(stdout);
			}
		} 
		if(magic == 3)
		{
			if ((i % (inputsize/160) == 0))
			{
				while ((picbuf = fgetc(pic)) != '\n')
				{
					if (picbuf == EOF) break;
					printf("%c", picbuf);
				}
				printf("\n");
			}
		}
		i++;
		j++;
	}
	printf("\n");

	fclose(inputfile_p);
	fclose(outputfile_p);
	free(key);
}

int main(int argc, char *argv[])
{
	int i;
	unsigned char key[4];
	unsigned char text[4];
	unsigned char crypted[4];
	unsigned char decrypted[4];

	char keyfile[PATH_MAX];
	int keylength;
	char inputfile[PATH_MAX];
	int inputlength;
	char outputfile[PATH_MAX];
	int outputlength;

	char keylengthbuf[LINE_MAX];
	
	if (argc > 1)
	{
		if (strcmp(argv[1], "keygen") == 0)
		{
			if (argc > 2)
			{
				strcpy(keyfile, argv[2]);
			}
			else
			{
				strcpy(keyfile, "key");
			}
			
			printf("Key-Laenge in Byte [128]: ");
			fgets(keylengthbuf, 10, stdin);
			keylength = atoi(keylengthbuf);
			if (keylength <= 0) 
			{
				keylength = 128;
			}
			keygen(keylength, keyfile);
			printf("%i Byte Key erfolgreich erzeugt (%s)\n", keylength, keyfile);
			return EXIT_SUCCESS;
		}
		
		if (argc != 5)
		{
			fprintf(stderr, "Usage: ./xorcrypt keygen [keyfilename]\n"); 
			fprintf(stderr, "                  -c[rypt] inputfile keyfile outputfile\n"); 
			fprintf(stderr, "                  -d[ecrypt] inputfile keyfile outputfile\n");
			return EXIT_FAILURE;
		}
		else
		{
			strcpy(inputfile, argv[2]);
			strcpy(keyfile, argv[3]);
			strcpy(outputfile, argv[4]);

			if (strcmp(inputfile, outputfile) == 0)
			{
				fprintf(stderr, "Err: inputfile == outputfile\n");
				return EXIT_FAILURE;
			}

			if (strcmp(keyfile, outputfile) == 0)
			{
				fprintf(stderr, "Err: keyfile == outputfile\n");
				return EXIT_FAILURE;
			}
		}

		if ((strcmp(argv[1], "-crypt") == 0) || (strcmp(argv[1], "-c") == 0))
		{
			crypt_file(inputfile, keyfile, outputfile, CRYPT);
			printf("%s successfully crypted: %s\n", inputfile, outputfile);
			return EXIT_SUCCESS;
		}

		if ((strcmp(argv[1], "-decrypt")) || (strcmp(argv[1], "-d") == 0))
		{
			crypt_file(inputfile, keyfile, outputfile, DECRYPT);
			printf("%s successfully decrypted: %s\n", inputfile, outputfile);
			return EXIT_SUCCESS;
		}
	}
	
	
	fprintf(stderr, "Usage: ./xorcrypt keygen [keyfilename]\n"); 
	fprintf(stderr, "                  crypt inputfile keyfile outputfile\n"); 
	fprintf(stderr, "                  decrypt inputfile keyfile outputfile\n");
			
	printf("-- Testmodus --\n");
	printf("Zu verschluesselnder Text (4 Zeichen + ENTER): ");
	for(i = 0; i < 4; i++)
	{
		text[i] = getchar();
	}
	while (getchar() != '\n')
	{
		/* zu langes keyword abfangen */
	}
	
	printf("Verschluesselungskey: (4 Zeichen + ENTER): ");
	for(i = 0; i < 4; i++)
	{
		key[i] = getchar();
	}
	fflush(stdin);

	/* Verschluesselung byte-weise */
	for(i = 0; i < 4; i++)
	{
		crypt_byte(text[i], key[i], &crypted[i]);
	}
	
	printf("Text: ");
	for (i = 0; i < 4; i++)
	{
		printf("%c", text[i]);
	}
	printf("\n");

	printf("Key: ");
	for (i = 0; i < 4; i++)
	{
		printf("%c", key[i]);
	}
	printf("\n");

	printf("Key (hex): ");
	for (i = 0; i < 4; i++)
	{
		printf("%0#x ", key[i]);
	}
	printf("\n");


	printf("Crypted (hex): ");
	for (i = 0; i < 4; i++)
	{
		printf("%0#x ", crypted[i]);
	}
	printf("\n");

	for (i = 0; i < 4; i++)
	{
		decrypt_byte(crypted[i], key[i], &decrypted[i]);
	}

	printf("Decrypted: ");
	for (i = 0; i < 4; i++)
	{
		printf("%c", decrypted[i]);
	}
	printf("\n");

	return EXIT_SUCCESS;
}
