// $Date: 2009-09-25 16:32:11 +0200 (Fri, 25 Sep 2009) $, $Revision: 2224 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/* INCLUDES */

#include "user/appl.h"
#include "device/cgastr.h"
#include "syscall/guarded_organizer.h"
#include "syscall/guarded_semaphore.h"
#include "object/debug.h"
#include "guard/secure.h"
#include "meeting/semaphore.h"
#include "device/keyboard.h"
#include "syscall/guarded_buzzer.h"

Guarded_Semaphore s(1);

/* GLOBALE VARIABLEN */
extern CGA_Stream kout;
extern Guarded_Organizer organizer;
extern Application application3;
extern Keyboard keyboard;
extern Guarded_Buzzer buzzer;

void Application::action () {

		while(1);

		//ACHTUNG, tut NIX!!


		int i = 0;

		if (ID == 6) {
				while (1) {
					Key k = keyboard.getkey();
					{ Secure s;
						kout.setpos(79, 0);
						kout << k.ascii() << endl;;
					}
				}
		}
/*		if (ID == 5) {
				while (1) {
					Key k = keyboard.getkey();
					{ Secure s;
						kout.setpos(78, 0);
						kout << k.ascii() << endl;;
					}
				}
		}
*/
		while(i < 10000)
		{ 
				if (ID == 1) {
						buzzer.set(250);	
						buzzer.sleep();
						{ Secure s;
							kout.setpos(40, 5);
							kout << "aufgewacht" << endl;
						}
				}
				{ Secure s;
						kout.setpos(0, ID);
						switch ((int)system.getCPUID()) {
								case 0:
										kout.setColor(CGA_Stream::BLUE, CGA_Stream::WHITE);
										break;
								case 1:
										kout.setColor(CGA_Stream::RED, CGA_Stream::WHITE);
										break;
								case 2:
										kout.setColor(CGA_Stream::GREEN, CGA_Stream::WHITE);
										break;
								case 3:
										kout.setColor(CGA_Stream::MAGENTA, CGA_Stream::WHITE);
										break;
						}
						kout << "Application " << ID << ": " << i << endl;
				}
				i++;
				
/*				if (i == 3000) {
						if (ID == 7) {
								DBG << "on CPU" << (int)system.getCPUID() << ": app7 exited"<< endl;
								organizer.exit();
						}
						if (ID == 4) {
								DBG << "on CPU" << (int)system.getCPUID() << ": app4 kills app3"<< endl;
								organizer.kill(application3);
						}
				} */
		}
		organizer.exit();
}

