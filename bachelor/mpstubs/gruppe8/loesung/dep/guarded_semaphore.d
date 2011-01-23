build/guarded_semaphore.o: syscall/guarded_semaphore.cc \
  syscall/guarded_semaphore.h guard/secure.h guard/guard.h \
  machine/spinlock.h object/queue.h object/chain.h guard/gate.h \
  guard/locker.h meeting/semaphore.h meeting/waitingroom.h \
  thread/customer.h thread/entrant.h thread/coroutine.h machine/toc.h
