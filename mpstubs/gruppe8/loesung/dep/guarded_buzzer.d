build/guarded_buzzer.o: syscall/guarded_buzzer.cc \
  syscall/guarded_buzzer.h meeting/buzzer.h meeting/waitingroom.h \
  object/queue.h object/chain.h meeting/bell.h meeting/bellringer.h \
  object/list.h syscall/guarded_organizer.h thread/organizer.h \
  thread/customer.h thread/entrant.h thread/coroutine.h machine/toc.h \
  thread/scheduler.h thread/dispatch.h machine/apicsystem.h \
  machine/mp_registers.h syscall/thread.h guard/secure.h guard/guard.h \
  machine/spinlock.h guard/gate.h guard/locker.h
