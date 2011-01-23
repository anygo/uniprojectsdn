// $Date: 2009-11-27 17:14:54 +0100 (Fri, 27 Nov 2009) $, $Revision: 2366 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

#ifndef __idle_thread_h__
#define __idle_thread_h__

/*! \file 
 *  \brief Enthaelt die Klasse IdleThread
 */

#include "thread/customer.h"
#include "syscall/guarded_organizer.h"
#include "object/debug.h"

/*! \brief Prozess, der immer dann laeuft, wenn eine CPU nichts zu tun hat. 
 * 
 *  In OOStuBS ist es nicht notwendig IdleThread zu verwenden. Eine 
 *  Implementierung ueber einen einfachen Idle-Loop im Scheduler ist dort 
 *  einfacher. 
 *  
 *  In MPStuBS hingegen vereinfacht die Verwendung von IdleThread die 
 *  Behandlung von "daeumchendrehenden" Prozessoren.
 *  \note
 *  Instanzen von IdleThread sollten nie in der Bereitliste des Schedulers 
 *  auftauchen, sondern immer getrennt gehalten werden, da sie ja nur dann 
 *  ausgefuehrt werden sollen, wenn kein normaler Thread mehr bereit ist.   
 * 
 */

extern Guarded_Organizer organizer;

class IdleThread : public Customer
{
public:
	IdleThread(void *tos) : Customer(tos) {}
	inline virtual void action() {
		while (true) {
			DBG << "IdleThread" << endl;
			organizer.sleep_until_IRQ();
		}
	}
};

#endif // __idle_thread_h__
