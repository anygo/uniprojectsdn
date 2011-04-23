/**
 * @file   malloc-test.c
 * @brief  A testcase for malloc(), calloc(), realloc() and free().
 * @author Christoph Erhardt <christoph.erhardt@informatik.stud.uni-erlangen.de>
 * @date   2008-11-23
 */



/* TODO:
 * - Add extensive tests trying to overwrite data and then reading the raw memory.
 * - Try to stop valgrind from complaining if the program is linked against the glibc versions of malloc() &
 *   Co. instead of the custom implementations.
 */



#include "halde.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/wait.h>



/** Number of bytes managed by malloc() (= 1 MiB). */
#define MANAGED_SIZE 1048576

/** Constant value used to mark an allocated memory block. */
#define OCCUPIED_BLOCK_MAGIC 0xcafebabe

/** Size of a block management structure. */
#define STRUCT_SIZE (sizeof(size_t) + sizeof(void *))

/** Size of a block for testing purposes. */
#define TEST_BLOCK_SIZE (STRUCT_SIZE * 8)



#ifndef _WIN32
	#define FORK_SUPPORTED 1
#else
	#define FORK_SUPPORTED 0
	#define abort myAbort
#endif



/** Statistics & settings. */
struct Statistics {
	unsigned int testCount;
	unsigned int successCount;
	unsigned int crashCount;
	int          doFork;
};



/** Indicates whether abort() is called on intention or accidentally. */
static int gs_abortIntended = 0;



/**
 * @brief Prints a headline.
 * @param string The headline.
 */
static void printHeadline(const char string[]) {
	putchar('\n');
	puts(string);
	puts("-------------");
}



/**
 * @brief Displays an error message and terminates the process.
 * @param description An error description.
 */
static void fatalError(const char description[]) {
	puts("FEHLGESCHLAGEN!");
	fputs("  ", stderr);
	perror(description);
	exit(EXIT_FAILURE);
}



/**
 * @brief Gets the name of a signal.
 * @param signal The signal.
 * @return The name of the signal.
 */
static const char *getSignalName(int signal) {
	switch (signal) {
#ifdef SIGABRT
	case SIGABRT:   return "SIGABRT";
	case SIGALRM:   return "SIGALRM";
	case SIGBUS:    return "SIGBUS";
	case SIGCHLD:   return "SIGCHLD";
	case SIGCONT:   return "SIGCONT";
	case SIGFPE:    return "SIGFPE";
	case SIGHUP:    return "SIGHUP";
	case SIGILL:    return "SIGILL";
	case SIGINT:    return "SIGINT";
	case SIGKILL:   return "SIGKILL";
	case SIGPIPE:   return "SIGPIPE";
	case SIGQUIT:   return "SIGQUIT";
	case SIGSEGV:   return "SIGSEGV";
	case SIGSTOP:   return "SIGSTOP";
	case SIGTERM:   return "SIGTERM";
	case SIGTSTP:   return "SIGTSTP";
	case SIGTTIN:   return "SIGTTIN";
	case SIGTTOU:   return "SIGTTOU";
	case SIGUSR1:   return "SIGUSR1";
	case SIGUSR2:   return "SIGUSR2";
	case SIGPOLL:   return "SIGPOLL";
	case SIGPROF:   return "SIGPROF";
	case SIGSYS:    return "SIGSYS";
	case SIGTRAP:   return "SIGTRAP";
	case SIGURG:    return "SIGURG";
	case SIGVTALRM: return "SIGVTALRM";
	case SIGXCPU:   return "SIGXCPU";
	case SIGXFSZ:   return "SIGXFSZ";
#endif
	default:        return "(unbekannt)";
	}
}



/**
 * @brief Runs a test.
 *
 * Every test is run in a separate child process so that a single error does not crash the whole program or
 * influence subsequent tests. If you want to examine a crash, you can disable forking by running the program
 * with the command line argument "--no-fork" (or "-n").
 *
 * @param stats       Pointer to the statistics structure.
 * @param function    Pointer to the testing function.
 * @param description Description of the test (or NULL, if none).
 * @param verbose     1 to display a success message, 0 to suppress it.
 */
static void runTest(struct Statistics *stats, int (*function)(void), const char description[], int verbose) {

	int success;

	fputs("* ", stdout);
	if (description != NULL) printf("%s: ", description);
	fflush(stdout);
	++stats->testCount;

	if (stats->doFork) {

		pid_t pid = fork();

		if (pid == -1) {       /* Error */

			fatalError("fork()");

		} else if (pid == 0) { /* Child process */

			exit(function() ? EXIT_SUCCESS : EXIT_FAILURE);

		} else {               /* Parent process */

			int status;

			if (wait(&status) == -1) fatalError("wait()");
			if (WIFEXITED(status) && (WEXITSTATUS(status) == EXIT_SUCCESS)) {
				success = 1;
			} else {
				if (WIFSIGNALED(status)) {
					printf("Absturz mit Signal %s\n", getSignalName(WTERMSIG(status)));
					++stats->crashCount;
				}
				success = 0;
			}
		}

	} else {

		success = function();
	}

	if (success) {
		++stats->successCount;
		if (verbose) puts("[OK]");
	}
}



/**
 * @brief Checks whether an impossible allocation has succeeded unexpectedly.
 * @param pointer The pointer. It is expected to be NULL and errno is expected to be ENOMEM.
 * @param message A message displayed if the pointer is not NULL.
 * @return 1 if pointer == NULL and errno == ENOMEM, false otherwise.
 */
static int checkError(void *pointer, const char message[]) {
	if (pointer != NULL) {
		puts((message != NULL) ? message : "Es wurde nicht NULL zurueckgeliefert!");
		free(pointer);
		return 0;
	} else if (errno != ENOMEM) {
		puts("errno muss im Fehlerfall auf ENOMEM gesetzt werden!");
		return 0;
	} else {
		return 1;
	}
}



/**
 * @brief Allocates a single big block that fills the 1 MiB completely.
 * @return 1 on success, 0 on error.
 */
static int singleBlockTest(void) {

	char        *p;
	char        *q;
	unsigned int size = MANAGED_SIZE - STRUCT_SIZE;

	printf("%u Bytes am Stueck belegen: ", size);
	fflush(stdout);

	p = malloc(size);
	if (p == NULL) {
		size -= STRUCT_SIZE;
		puts("fehlgeschlagen");
		printf("Zumindest %u Bytes muesste man aber holen koennen: ", size);
		fflush(stdout);
		p = malloc(size);
		if (p == NULL) {
			puts("FEHLGESCHLAGEN!");
			return 0;
		}
	}

	memset(p, 0, size);
	errno = 0;
	q     = malloc(TEST_BLOCK_SIZE);
	free(p);

	return checkError(q, "Ein anschliessendes malloc() war erfolgreich?!?");
}



/**
 * @brief Allocates blocks with random sizes that fill the 1 MiB completely.
 * @return 1 on success, 0 on error.
 */
static int multiBlockTest(void) {

	char  *p[MANAGED_SIZE / (STRUCT_SIZE * 2)];
	float  minSize   = TEST_BLOCK_SIZE *    4;
	float  maxSize   = TEST_BLOCK_SIZE * 1024;
	size_t i         = 0;
	int    success   = 1;
	int    freeBytes = MANAGED_SIZE - STRUCT_SIZE;

	srand((unsigned int) time(NULL));

	do {
		int blockSize = 1 + (size_t) (maxSize * rand() / (RAND_MAX + minSize));

		if (blockSize > freeBytes) blockSize = freeBytes;
		p[i] = malloc(blockSize);
		if (p[i] == NULL) {
			puts("FEHLGESCHLAGEN!");
			printf("malloc() schlug fehl, obwohl rein rechnerisch noch mindestens %u Bytes frei sein "
			       "muessten: %s\n\n", freeBytes, strerror(ENOMEM));
			++i;
			success = 0;
			break;
		}
		memset(p[i], 0, blockSize);
		++i;
		freeBytes -= ((2 * STRUCT_SIZE) + blockSize - (blockSize % STRUCT_SIZE));
	} while (freeBytes > 0);

	do {
		free(p[--i]);
	} while (i > 0);
	return success;
}



/**
 * @brief Tests the behavior of malloc() when 0 is passed as the size.
 * @return 1 on success, 0 on error.
 */
static int mallocZeroTest(void) {

	char       *p        = malloc(0);
	const char *breakVal = sbrk(0);

	if (breakVal == (char *) -1) fatalError("sbrk()");

	if ((p != NULL) && ((p + MANAGED_SIZE < breakVal + STRUCT_SIZE) || (p >= breakVal))) {
		puts("Zurueckgelieferter Zeiger ist ungueltig!");
		return 0;
	}

	free(p);
	return 1;
}



/**
 * @brief Tries to allocate too many bytes.
 * @return 1 on success, 0 on error.
 */
static int mallocFailTest(void) {

	unsigned int size = MANAGED_SIZE * 2;

	printf("%u Bytes belegen (unerfuellbar): ", size);
	fflush(stdout);
	errno = 0;
	return checkError(malloc(size), NULL);
}



/**
 * @brief Calculates the alignment of the address pointers returned by malloc().
 * @return 1 on success, 0 on error.
 */
static int alignmentTest(void) {

	size_t       i;
	char        *p[STRUCT_SIZE];
	size_t       size      = STRUCT_SIZE;
	unsigned int alignment = STRUCT_SIZE;

	/* Allocate blocks of different sizes, incremented by one (STRUCT_SIZE <= size <= 2 * STRUCT_SIZE).
	 * The alignment is the GCD of all block addresses.
	 */
	for (i = 0; i < STRUCT_SIZE; ++i) {
		p[i] = malloc(++size);
		if (p[i] == NULL) fatalError("malloc()");
		while (((size_t) p[i]) % alignment != 0) {
			alignment /= 2;
		}
	}
	for (i = 0; i < STRUCT_SIZE; ++i) {
		free(p[i]);
	}

	printf("%u Bytes ", alignment);
	if (alignment == STRUCT_SIZE) {
		puts("(OK)");
	} else if (alignment >= sizeof(void *)) {
		puts("(hervorragend)");
	} else {
		puts("(nicht SPARC-tauglich, aber OK)");
	}
	return 1;
}



/**
 * @brief Checks realloc() increasing the size of a block.
 * @return 1 on success, 0 on error.
 */
static int reallocIncreaseTest(void) {

	size_t i;
	char  *q;
	size_t newSize = TEST_BLOCK_SIZE * 2;
	int    success = 1;
	char  *p       = malloc(TEST_BLOCK_SIZE);

	if (p == NULL) fatalError("malloc");

	/* Fill block with data */
	for (i = 0; i < TEST_BLOCK_SIZE; ++i) {
		p[i] = i;
	}

	q = realloc(p, newSize);
	if (q == NULL) fatalError("realloc()");

	/* Check if contents of block have been preserved */
	for (i = 0; i < TEST_BLOCK_SIZE; ++i) {
		if (q[i] != i) break;
	}
	if (i < TEST_BLOCK_SIZE) {
		puts("Inhalt des Blocks wurde nicht kopiert!");
		success = 0;
	} else if (q == p) {
		fputs("Overkill, aber ", stdout);
	}

	memset(q, 0, newSize);
	free(q);
	return success;
}



/**
 * @brief Checks realloc() reducing the size of a block.
 * @return 1 on success, 0 on error.
 */
static int reallocDecreaseTest(void) {

	size_t i;
	char  *q;
	char  *p       = malloc(TEST_BLOCK_SIZE);
	size_t newSize = TEST_BLOCK_SIZE / 2;
	int    success = 1;

	if (p == NULL) fatalError("malloc");

	/* Fill block with data */
	for (i = 0; i < TEST_BLOCK_SIZE; ++i) {
		p[i] = i;
	}

	q = realloc(p, newSize);
	if (q == NULL) fatalError("realloc()");

	/* Check if contents of block have been preserved */
	for (i = 0; i < newSize; ++i) {
		if (q[i] != i) break;
	}
	if (i < newSize) {
		puts("Inhalt des Blocks wurde nicht kopiert!");
		success = 0;
	} else if (q == p) {
		fputs("elegant & ", stdout);
	}

	memset(q, 0, newSize);
	free(q);
	return success;
}



/**
 * @brief Tests the behavior of realloc() when it is passed a NULL pointer.
 * @return 1 on success, 0 on error.
 */
static int reallocNullTest(void) {

	char *p = realloc(NULL, TEST_BLOCK_SIZE);

	if (p == NULL) fatalError("realloc()");

	memset(p, 0, TEST_BLOCK_SIZE);
	free(p);
	return 1;
}



/**
 * @brief Tests the behavior of realloc() when 0 is passed as the size.
 * @return 1 on success, 0 on error.
 */
static int reallocZeroTest(void) {

	char *p = malloc(TEST_BLOCK_SIZE);

	if (p == NULL) fatalError("malloc()");

	p = realloc(p, 0);
	if (p != NULL) free(p);
	return 1;
}



/**
 * @brief Tries to resize an existing memory block too far.
 * @return 1 on success, 0 on error.
 */
static int reallocFailTest(void) {

	char        *p;
	unsigned int size = MANAGED_SIZE * 2;

	printf("Block auf %u Bytes vergroessern (unerfuellbar): ", size);
	fflush(stdout);

	p = malloc(TEST_BLOCK_SIZE);
	if (p == NULL) fatalError("malloc()");

	errno = 0;
	if (checkError(realloc(p, size), NULL) && (*(((char **) p) - 1) != (char *) OCCUPIED_BLOCK_MAGIC)) {
		puts("Alter Block darf im Fehlerfall nicht freigegeben werden!");
		return 0;
	} else {
		return 1;
	}
}



/**
 * @brief Overwrites a block management structure and then calls realloc().
 * @return 1 on success, 0 on error.
 */
static int reallocCorruptTest(void) {

	char *p = malloc(TEST_BLOCK_SIZE);

	if (p == NULL) fatalError("malloc()");

	memset(p - STRUCT_SIZE, 0, STRUCT_SIZE);
	gs_abortIntended = 1;
	p = realloc(p, 2 * TEST_BLOCK_SIZE);

	puts("abort() wurde nicht aufgerufen!");
	free(p);
	return 0;
}



/**
 * @brief Checks whether calloc() fills the allocated block with zeros.
 * @return 1 on success, 0 on error.
 */
static int callocTest(void) {

	size_t i;
	int    success = 1;
	char  *p       = calloc(32, 2);

	if (p == NULL) fatalError("calloc()");

	for (i = 0; i < 64; ++i) {
		if (p[i] != 0) break;
	}
	if (i != 64) {
		puts("Speicher wurde nicht ausgenullt!");
		success = 0;
	}
	free(p);
	return success;
}



/**
 * @brief Tries to allocate too many bytes.
 * @return 1 on success, 0 on error.
 */
static int callocFailTest(void) {
	printf("%u Bytes belegen (unerfuellbar): ", MANAGED_SIZE * 32);
	fflush(stdout);
	errno = 0;
	return checkError(calloc(MANAGED_SIZE, 32), NULL);
}



/**
 * @brief Tests the behavior of free(NULL).
 * @return 1 on success, 0 on error.
 */
static int freeNullTest(void) {
	free(NULL);
	return 1;
}



/**
 * @brief Overwrites a block management structure and then calls free().
 * @return 1 on success, 0 on error.
 */
static int freeCorruptTest(void) {

	char *p = malloc(TEST_BLOCK_SIZE);

	if (p == NULL) fatalError("malloc()");

	memset(p - STRUCT_SIZE, 0, STRUCT_SIZE);
	gs_abortIntended = 1;
	free(p);

	puts("abort() wurde nicht aufgerufen!");
	return 0;
}



/**
 * @brief Entry point.
 *
 * Run the program with "--no-fork" or "-n" to prevent it from forking. This might be useful for examining
 * crashes, but suppresses the corruption tests.
 *
 * @param argc Number of command line arguments.
 * @param argv Array with the command line arguments.
 * @return Always EXIT_SUCCESS.
 */
int main(int argc, char *argv[]) {

	struct Statistics s = {0, 0, 0, FORK_SUPPORTED};

	/* Print introduction */
	puts("=====================================");
	puts("MALLOC-TEST (Testcase fuer Aufgabe 4)");
	puts("=====================================");

	/* Check command line */
	if ((argc == 2) && ((strcmp(argv[1], "--no-fork") == 0) || (strcmp(argv[1], "-n") == 0))) {
		s.doFork = 0;
		putchar('\n');
		puts("fork()-Modus wurde zwecks Fehlersuche ausgeschaltet.");
		puts("Die abort()-Tests wurden deshalb deaktiviert.");
	}

	/* malloc() tests */
	printHeadline("1. malloc():");
	runTest(&s, singleBlockTest, NULL,                                             1);
	runTest(&s, multiBlockTest, "Speicher mit mehreren Anfragen komplett fuellen", 1);
	runTest(&s, mallocZeroTest,  "malloc(0)",                                      1);
	runTest(&s, mallocFailTest,  NULL,                                             1);
	runTest(&s, alignmentTest,   "Alignment",                                      0);

	/* free() tests */
	printHeadline("2. free():");
	runTest(&s, freeNullTest, "free(NULL)", 1);
	if (s.doFork) runTest(&s, freeCorruptTest, "free() mit kaputter Verwaltungsstruktur", 0);

	/* realloc() tests */
	printHeadline("3. realloc():");
	runTest(&s, reallocIncreaseTest, "Block mit realloc() vergroessern",   1);
	runTest(&s, reallocDecreaseTest, "Block mit realloc() verkleinern",    1);
	runTest(&s, reallocNullTest,     "realloc() mit NULL-Zeiger aufrufen", 1);
	runTest(&s, reallocZeroTest,     "realloc() mit Groesse 0",            1);
	runTest(&s, reallocFailTest,     NULL,                                 1);
	if (s.doFork) runTest(&s, reallocCorruptTest, "realloc() mit kaputter Verwaltungsstruktur", 0);

	/* calloc() tests */
	printHeadline("4. calloc():");
	runTest(&s, callocTest,     "calloc()-Test", 1);
	runTest(&s, callocFailTest, NULL,            1);

	/* Print summary */
	putchar('\n');
	puts("=======================================");
	printf("%u von %u Tests waren erfolgreich. ", s.successCount, s.testCount);
	if (s.successCount == s.testCount) {                /* All tests successful */
		puts(":-D");
	} else if (s.successCount * 3 >= s.testCount * 2) { /* >= 2/3 of all tests successful */
		puts(":-)");
	} else if (s.successCount * 3 >= s.testCount) {     /* >= 1/3 of all tests successful */
		puts(":-|");
	} else if (s.successCount > 0) {                    /* At least one test successful */
		puts(":-/");
	} else {                                            /* Everything FUBAR */
		puts("=:-O");
	}
	puts("=======================================");

	/* Print debugging hint if a crash has occurred */
	if (s.crashCount > 0) {
		printf("\n%u Tests haben Abstuerze verursacht. Tipp fuers Debuggen:\n", s.crashCount);
		puts("    gdb ./malloc-test");
		puts("    run --no-fork");
	}

	return EXIT_SUCCESS;
}



/** Terminates the program. */
void abort(void) {
	if (gs_abortIntended == 0) {
		puts("FEHLGESCHLAGEN");
		puts("  abort() wurde unerwartet aufgerufen! :-(");
		exit(EXIT_FAILURE);
	} else {
		puts("[OK]");
		exit(EXIT_SUCCESS);
	}
}
