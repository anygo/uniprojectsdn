build/guarded_keyboard.o: syscall/guarded_keyboard.cc \
  syscall/guarded_keyboard.h device/keyboard.h machine/keyctrl.h \
  machine/io_port.h machine/key.h guard/gate.h object/chain.h \
  syscall/guarded_semaphore.h guard/secure.h guard/guard.h \
  machine/spinlock.h object/queue.h guard/locker.h meeting/semaphore.h \
  meeting/waitingroom.h thread/customer.h thread/entrant.h \
  thread/coroutine.h machine/toc.h
