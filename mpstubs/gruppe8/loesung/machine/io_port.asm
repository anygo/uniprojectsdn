; $Date: 2009-08-11 16:57:46 +0200 (Tue, 11 Aug 2009) $, $Revision: 2208 $, $Author: benjamin $
; kate: encoding ISO-8859-15;
; vim: set fileencoding=latin-9:
; -*- mode: c; coding: latin-9 -*-
 
;*****************************************************************************
;* Betriebssysteme                                                           *
;*---------------------------------------------------------------------------*
;*                                                                           *
;*                             I O _ P O R T                                 *
;*                                                                           *
;*---------------------------------------------------------------------------*
;* Die hier definierten Funktionen stellen die Maschinebefehle 'in' und      *
;* 'out' fuer die Klasse IO_Port zur Verfuegung.                             *
;*****************************************************************************

; EXPORTIERTE FUNKTIONEN

[GLOBAL outb]
[GLOBAL outw]
[GLOBAL inb]
[GLOBAL inw]

; IMPLEMENTIERUNG DER FUNKTIONEN

[SECTION .text]
	
; OUTB: Byteweise Ausgabe eines Wertes ueber einen I/O-Port.
;
;       C-Prototyp: void outb (int port, int value);

outb:
	push	ebp
	mov	ebp,esp
	mov	edx,[8+ebp]
	mov	eax,[12+ebp]
	out	dx,al
	pop	ebp
	ret

; OUTW: Wortweise Ausgabe eines Wertes ueber einen I/O-Port.
;
;       C-Prototyp: void outw (int port, int value);

outw:
	push	ebp
	mov	ebp,esp
	mov	edx,[8+ebp]
	mov	eax,[12+ebp]
	out	dx,ax
	pop	ebp
	ret

; INB: Byteweises Einlesen eines Wertes ueber einen I/O-Port.
;
;      C-Prototyp: int inb (int port);

inb:
	push	ebp
	mov	ebp,esp
	mov	edx,[8+ebp]
	xor	eax,eax
	in	al,dx
	pop	ebp
	ret

; INW: Wortweises Einlesen eines Wertes ueber einen I/O-Port.
;
;      C-Prototyp: int inw (int port);

inw:
	push	ebp
	mov	ebp,esp
	mov	edx,[8+ebp]
	xor	eax,eax
	in	ax,dx
	pop	ebp
	ret

