
#include "machine/spinlock.h"

Spinlock::Spinlock() {
	__sync_lock_release(lock_status);
}

void Spinlock::lock() {
	while ((__sync_lock_test_and_set(lock_status, 1)) == 1); 
}

void Spinlock::unlock() {
	__sync_lock_release(lock_status);
}
