; $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
; kate: encoding ISO-8859-15;
; vim: set fileencoding=latin-9:
; -*- coding: latin-9 -*-

;******************************************************************************
;* Betriebssysteme                                                            *
;*----------------------------------------------------------------------------*
;*                                                                            *
;*                             S E T U P  A P                                 *
;*                                                                            *
;*----------------------------------------------------------------------------*
;* Der Setup-Code liegt an 0x00004000. Er wird nach einem Startup-IPI noch im *
;* 'Real-Mode' gestartet, so dass zu Beginn auch noch BIOS-Aufrufe erlaubt    *
;* sind. Dann werden jedoch alle Interrupts verboten, die Adressleitung A20   *
;* aktiviert und die Umschaltung in den 'Protected-Mode' vorgenommen. Alles   *
;* weitere uebernimmt der Startup-Code des Systems.                           *
;* Dieser Code wird von SMPSystem::copySetupAPtoLowMem() reloziert!           *
;******************************************************************************

[GLOBAL setup_ap]
[EXTERN startup_ap]

[SECTION .setup_ap_seg]

USE16

;
; Segmentregister initialisieren
;

setup_ap:
	mov	ax,cs		; Daten- und Codesegment sollen
	mov	ds,ax		; hierher zeigen. Stack brauchen wir hier nicht.
;
; So, jetzt werden die Interrupts abgeschaltet
;
	cli			; Maskierbare Interrupts verbieten
	mov	al,0x80		; NMI verbieten
	out	0x70,al
;
; IDT und GDT setzen
;
	lidt	[idt_48 - setup_ap]
	lgdt	[gdt_48 - setup_ap]
;
; Umschalten in den Protected Mode
;
	mov	ax,1
	lmsw	ax
	jmp flush_instr

flush_instr:
	;jmp	dword 0x08:0x30000
	jmp	dword 0x08:startup_ap
	hlt

gdt:
	dw	0,0,0,0		; NULL Deskriptor

	dw	0xFFFF		; 4Gb - (0x100000*0x1000 = 4Gb)
	dw	0x0000		; base address=0
	dw	0x9A00		; code read/exec
	dw	0x00CF		; granularity=4096, 386 (+5th nibble of limit)

	dw	0xFFFF		; 4Gb - (0x100000*0x1000 = 4Gb)
	dw	0x0000		; base address=0
	dw	0x9200		; data read/write
	dw	0x00CF		; granularity=4096, 386 (+5th nibble of limit)

idt_48:
	dw	0		; idt limit=0
	dw	0,0		; idt base=0L
	
gdt_48:
	dw	0x18		; GDT Limit=24, 3 GDT Eintraege
	dd	0x40000 + gdt - setup_ap; Physikalische Adresse der GDT
