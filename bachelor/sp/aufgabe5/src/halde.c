#include "halde.h" 
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define MAGIC ((void*)0xcafebabe)

struct mblock
{
	size_t size;
	struct mblock *next;
};

static struct mblock *fsp = NULL;
static int verwaltung;

void *malloc(size_t size) {

	struct mblock *tmp;
	struct mblock *prev;
	struct mblock *new;
	int units;

	verwaltung = 1;

	/* Initialisierung */
	if (fsp == NULL)
	{
		char *newmem;
		newmem = (char *)sbrk(1024*1024);
		if ((void *)newmem == (void *)-1)
		{
			errno = ENOMEM;
			return NULL;
		}
		fsp = (struct mblock *)newmem;
		fsp->size = 1024*1024-sizeof(struct mblock);
		fsp->next = NULL;
	}
	/* abgeschlossen */

	if (size > 1024*1024 - sizeof(struct mblock))
	{
		errno = ENOMEM;
		return NULL;
	}


	tmp = fsp;
	prev = NULL;

	while (tmp->next != NULL)
	{
		if (size <= tmp->size && size + sizeof(struct mblock) > tmp->size)
		{
			verwaltung = 0; /* false -> keine Verwaltungsstruktur */
			break;
		}
		if (size + sizeof(struct mblock) > tmp->size)
		{
			prev = tmp;
			if (tmp != tmp->next)
			{
				tmp = tmp->next;
			}
			else
			{
				errno = ENOMEM;
				return NULL;
			}
		}
		else
			break;
	}

	if (size + sizeof(struct mblock) > tmp->size && verwaltung) /* Sonderfall ueberpruefen */
	{
		errno = ENOMEM;
		return NULL;
	}

	units = ((size - 1)/sizeof(struct mblock)) + 1;
	
	new = tmp + units + 1;

	if (verwaltung)
	{
		new->size = tmp->size - units*(sizeof(struct mblock)) - sizeof(struct mblock);
	}
	else
	{
		new->size = tmp->size - units*(sizeof(struct mblock));
	}

	new->next = tmp->next;

	tmp->size = units*sizeof(struct mblock);
	tmp->next = MAGIC;
	
	if (prev != NULL)
	{
		prev->next = new;
	}
	else
	{
		fsp = new;
	}

	return (void *)(tmp + 1);
}

void free(void* ptr) {
	struct mblock *mbp;

	if (ptr != NULL)
	{
		mbp = (struct mblock *)ptr - 1; /* 1x struct mblock abziehen */
		if (mbp->next == MAGIC)
		{
			mbp->next = fsp;
			fsp = mbp;
		}
		else
		{
			abort();
		}
	}
}

void *realloc(void *ptr,size_t size) {
	if (ptr == NULL)
	{
		return malloc(size);
	}

	if ((((struct mblock *)ptr)-1)->next == MAGIC)
	{
		void *vtmp;
		struct mblock *mtmp = (struct mblock *)ptr - 1;

		vtmp = (void *)malloc(size);
		if (vtmp == NULL)
		{
			/* errno wird in malloc bereits gesetzt */
			return NULL;
		}
		
		/* alten Speicher kopieren */
		memcpy(vtmp, ptr, mtmp->size);
		/* und anschliessend freigeben */
		free(ptr);
		/* jetzt noch den Zeiger auf den neuen Speicher zurueckgeben... */
		return vtmp;
	}

	abort();	
}

void *calloc(size_t nmemb, size_t size) {
	void *ret;

	ret = malloc(size*nmemb);
	
	if (ret == NULL)
	{
		/* errno wird bereits in malloc gesetzt */
		return NULL;
	}
	/* neu angeforderten Speicher mit 0en fuellen */
	memset(ret, 0, size*nmemb);

	/* und noch den Pointer auf den Speicher zurueckgeben... */
	return ret;
}
