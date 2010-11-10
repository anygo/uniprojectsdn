#ifndef HALDE_H
#define HALDE_H

#include <sys/types.h>

/*
   malloc() allocates size bytes and returns a pointer to the
   allocated memory. The memory is not cleared.

   RETURN VALUE: The value returned is a pointer
   to the allocated memory, which is suitably aligned for any
   kind of variable, or NULL if the request fails. The
   errno will be set to indicate the error.
*/
void *malloc(size_t);

/*
   free() frees the memory space pointed to by ptr, which
   must have been returned by a previous call to malloc(),
   calloc() or realloc(). Otherwise, or if free(ptr) has
   already been called before the program is aborted.
   If ptr is NULL, no operation is performed.

   RETURN VALUE: no value
*/
void free(void*);

/*
   realloc()  changes the size of the memory block pointed to
   by ptr to size bytes.  The contents will be  unchanged  to
   the minimum of the old and new sizes; newly allocated mem­
   ory will be uninitialized.  If ptr is NULL,  the  call  is
   equivalent  to malloc(size). Unless ptr is  NULL,  it must
   have  been  returned by an earlier call to malloc(),
   calloc() or realloc().

   RETURN VALUE: The value returned is a pointer
   to the allocated memory, which is suitably aligned for any
   kind of variable, or NULL if the request fails. The
   errno will be set to indicate the error.
*/
void *realloc(void*,size_t);

/*
   calloc() allocates memory for an array of nmemb elements
   of size bytes each and returns a pointer to the allocated
   memory. The memory is set to zero.

   RETURN VALUE: The value returned is a pointer
   to the allocated memory, which is suitably aligned for any
   kind of variable, or NULL if the request fails. The
   errno will be set to indicate the error.
*/
void *calloc(size_t,size_t);

#endif
