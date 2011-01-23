// $Date: 2009-09-25 16:32:11 +0200 (Fri, 25 Sep 2009) $, $Revision: 2224 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file
 *  \brief Enthält mit den main() und main_ap() Funktionen den Startpunkt für 
 *  das System
 */

/* INCLUDES */
#include "device/cgastr.h"
#include "device/panic.h"
#include "device/watch.h"
#include "guard/guard.h"
#include "guard/locker.h"
#include "guard/secure.h"
#include "machine/apicsystem.h"
#include "machine/cpu.h"
#include "machine/ioapic.h"
#include "machine/lapic.h"
#include "meeting/bellringer.h"
#include "object/debug.h"
#include "syscall/guarded_buzzer.h"
#include "syscall/guarded_keyboard.h"
#include "syscall/guarded_organizer.h"
#include "syscall/guarded_vesagraphics.h"
#include "thread/assassin.h"
#include "thread/idlethread.h"
#include "thread/wakeup.h"
#include "user/appl.h"
#include "user/game.h"
#include "user/santas_house.h"

/* Globale Variablen */
Assassin assassin;
Bellringer bellringer;
CGA_Stream dall(0, 79, 15, 24);
CGA_Stream dout_CPU0(1, 37, 16, 19);
CGA_Stream dout_CPU1(42, 78, 16, 19);
CGA_Stream dout_CPU2(1, 37, 21, 24);
CGA_Stream dout_CPU3(42, 78, 21, 24);
CGA_Stream intro(0, 79, 0, 2);
CGA_Stream kout(0, 79, 2, 14, true);
CPU cpu;
Guard guard;
Guarded_Buzzer buzzer;
Guarded_Keyboard keyboard;
Guarded_Organizer organizer;
IOAPIC ioapic;
Keyboard_Controller key_controller; 
Panic panic;
Plugbox plugbox;
WakeUp wakeup;
Watch watch(15000);
char drehding[CPU_MAX];
char buf1[1024*1024*4];
char buf2[1024*1024*4];
Guarded_VESAGraphics vesa(buf1, buf2);
extern APICSystem system;
extern LAPIC::LAPIC lapic;

static unsigned char stack[1024];
Game game( (void *)(stack+(sizeof(stack))));


static unsigned char idleStack0[1024];
static unsigned char idleStack1[1024];
static unsigned char idleStack2[1024];
static unsigned char idleStack3[1024];
static unsigned char idleStack4[1024];
static unsigned char idleStack5[1024];
static unsigned char idleStack6[1024];
static unsigned char idleStack7[1024];
IdleThread myIdleThread0((void *) (idleStack0 + (sizeof(idleStack0))));
IdleThread myIdleThread1((void *) (idleStack1 + (sizeof(idleStack1))));
IdleThread myIdleThread2((void *) (idleStack2 + (sizeof(idleStack2))));
IdleThread myIdleThread3((void *) (idleStack3 + (sizeof(idleStack3))));
IdleThread myIdleThread4((void *) (idleStack4 + (sizeof(idleStack4))));
IdleThread myIdleThread5((void *) (idleStack5 + (sizeof(idleStack5))));
IdleThread myIdleThread6((void *) (idleStack6 + (sizeof(idleStack6))));
IdleThread myIdleThread7((void *) (idleStack7 + (sizeof(idleStack7))));


static const unsigned long CPU_STACK_SIZE = 256;
static unsigned long cpu_stack[(CPU_MAX - 1) * CPU_STACK_SIZE]; 

/*! \brief Einsprungpunkt ins System
 */
extern "C" int main() {
    

	ioapic.init();

	organizer.set_idle_thread(system.getCPUID(), &myIdleThread0);

	vesa.init();
	VBEModeData_t *mode = vesa.find_mode(1024, 768, 24);
	vesa.set_mode(mode);

	wakeup.activate();

	organizer.ready(game);
	
	guard.enter();
	cpu.disable_int();	
    
	APICSystem::SystemType type = system.getSystemType();
    unsigned int numCPUs = system.getNumberOfCPUs();
    DBG << "Is SMP system? " << (type == APICSystem::MP_APIC) << endl;
    DBG << "Number of CPUs: " << numCPUs << endl;
    switch (type) {
    case APICSystem::MP_APIC:
    {
        //Startet die AP-Prozessoren
        for (unsigned int i = 1; i < numCPUs; i++) {
            void* startup_stack = (void *) &(cpu_stack[(i) * CPU_STACK_SIZE]);

            system.bootCPU(i, startup_stack);
        }

    }
    case APICSystem::UP_APIC:
    {

    break;
    }
    case APICSystem::UNDETECTED: 
    {
    }
    }
   
	watch.windup();
	assassin.hire();
	keyboard.plugin();
	cpu.enable_int();
	organizer.Organizer::schedule();

	return 0;
}

/*! \brief Einsprungpunkt für Applikationsprozessoren
 */
extern "C" int main_ap() {
//    DBG << "CPU " << (int)system.getCPUID() << " is up" << endl;
	system.waitForCallout();
    system.initLAPIC();
    system.callin();

	switch (system.getCPUID()) 
	{
		case 1: organizer.set_idle_thread(system.getCPUID(), &myIdleThread1);
				break;
		case 2: organizer.set_idle_thread(system.getCPUID(), &myIdleThread2);
				break;
		case 3: organizer.set_idle_thread(system.getCPUID(), &myIdleThread3);
				break;
		case 4: organizer.set_idle_thread(system.getCPUID(), &myIdleThread4);
				break;
		case 5: organizer.set_idle_thread(system.getCPUID(), &myIdleThread5);
				break;
		case 6: organizer.set_idle_thread(system.getCPUID(), &myIdleThread6);
				break;
		case 7: organizer.set_idle_thread(system.getCPUID(), &myIdleThread7);
				break;
	}

	guard.enter();
	cpu.enable_int();
	organizer.Organizer::schedule();
    
	return 0;
}

