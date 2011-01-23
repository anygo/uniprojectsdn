// $Date: 2009-12-04 17:29:35 +0100 (Fri, 04 Dec 2009) $, $Revision: 2382 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*-

/*! \file
 *  \brief Enthaelt die Struktur struct toc.
 */

#ifndef __toc_include__
#define __toc_include__

/*! \brief Die Struktur toc dient dazu, bei einem Koroutinenwechsel die Werte 
 *  der nicht-fluechtigen Register zu sichern.
 *  
 *  Beim GNU C Compiler sind eax, ecx und edx fluechtige Register, die bei 
 *  Funktionsaufrufen und somit auch bei einem Koroutinenwechsel keine spaeter
 *  noch benoetigten Werte haben duerfen. Daher muss in der Struktur toc auch
 *  kein Platz fuer sie bereitgestellt werden.
 *  
 *  Achtung: Fuer den Zugriff auf die Elemente von struct toc aus einer
 *  Assemblerfunktion heraus werden in der Datei toc.inc Namen fuer die
 *  benoetigten Abstaende der einzelnen Elemente zum Anfang der Struktur
 *  definiert. Damit dann auch auf die richtigen Elemente zugegriffen wird,
 *  muessen sich die Angaben von toc.h und toc.inc exakt entsprechen. Wer also
 *  toc.h aendert, muss auch toc.inc anpassen (und umgekehrt).   
 */ 
struct toc {
    void *ebx;
    void *esi;
    void *edi;
    void *ebp;
    void *esp;
};

class Coroutine;

void toc_settle(struct toc *regs, void *tos, void (*kickoff) (Coroutine *), Coroutine *object);
extern "C" {
	void toc_go(struct toc *regs);
	void toc_switch(struct toc *regs_now, struct toc *regs_then);
}

#endif

