#ifndef PLIST_H
#define PLIST_H

#include <sys/types.h>

/** \file plist.h
 *
 *  \brief Linked List for maintaining process id - command line pairs.
 */

/**
 *  \brief Inserts a new pid-command line pair into the linked list.
 *
 * During the insert operation, the passed commandLine is copied to
 * an internally allocated buffer. The caller may free or otherwise
 * reuse the memory occupied by commandLine after return from
 * insertElement.
 *
 *  \param pid The process id of the pair that is to be inserted.
 *  \param commandLine The commandLine corresponding to the process with id pid.
 *
 *  \return pid on success, negative value on error
 *    \retval pid  success
 *    \retval  -1  a pair with the given pid already exists
 *    \retval  -2  insufficient memory to complete the operation
 */
int insertElement(pid_t pid, const char *commandLine);

/**
 *  \brief Remove a specific pid-command line pair from the linked list.
 *
 * The linked list is searched for a pair with the given pid. If such a pair is
 * found, the '\\0'-terminated command line is copied to the buffer provided by
 * the caller. If the length of the command line exceeds the size of the
 * buffer, only the first (buffersize-1) characters of the command line are
 * copied to the buffer and terminated with the '\\0' character. Upon
 * completion, removeElement deallocates all resources that were occupied by
 * the removed pair.
 *
 *  \param pid The process id of the pair that is to be removed.
 *  \param commandLineBuffer A buffer provided by the caller that the '\\0'-terminated commandline is written to.
 *  \param bufferSize The size of commandLineBuffer.
 *
 *  \return actual length of the command line on success, negative value on error.
 *    \retval >0  success, actual length of the command line
 *    \retval -1  a pair with the given pid does not exist
 */
int removeElement(pid_t pid, char *commandLineBuffer, size_t bufferSize);

/**
 *  \brief Reset the list iterator position to the head of the list.
 *
 * This function should be invoked before starting a list traversal with
 * the nextElement() function. The internal position of the list iterator
 * is reset to the head of the list.
 */
void rewindPList();

/**
 *  \brief Provides the next element in the list and advances the list iterator.
 *
 * This functions returns the command line and the pid of the current position of
 * the list iterator, and advances the iterator to the next element afterwards.
 * The function provides a pointer to the internal storage allocated for the
 * commandLine. This storage must not be used after the element was removed
 * using the removeElement function. The implementation of nextElement is not
 * thread safe. A list traversal must not be interleaved by insertElement or
 * removeElement operations.
 *
 *  \param cmdLine Pointer to a location, where a pointer to the interal storage location of command line can be written to.
 *  \param pid     Pointer to a location suitable to store a pid value.
 *
 *  \return 0 on success, negative value when the iterator has reached the end of the list  
 */
int nextElement(const char **cmdLine, pid_t *pid);


#endif

