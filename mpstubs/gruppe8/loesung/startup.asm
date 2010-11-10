; $Date: 2009-10-23 14:21:48 +0200 (Fri, 23 Oct 2009) $, $Revision: 2275 $, $Author: benjamin $
; kate: encoding ISO-8859-15;
; vim: set fileencoding=latin-9:
; -*- coding: latin-9 -*-

;******************************************************************************
;* Betriebssysteme                                                            *
;*----------------------------------------------------------------------------*
;*                                                                            *
;*                        S T A R T U P . A S M                               *
;*                                                                            *
;*----------------------------------------------------------------------------*
;* 'startup' ist der Eintrittspunkt des eigentlichen Systems. Die Umschaltung *
;* in den 'Protected Mode' ist bereits erfolgt. Es wird alles vorbereitet,    *
;* damit so schnell wie moeglich die weitere Ausfuehrung durch C-Code erfol-  *
;* gen kann.                                                                  *
;******************************************************************************

; Multiboot-Konstanten
MULTIBOOT_PAGE_ALIGN	equ	1<<0
MULTIBOOT_MEMORY_INFO	equ	1<<1

; Magic-Number fuer Multiboot
MULTIBOOT_HEADER_MAGIC	equ	0x1badb002
; Multiboot-Flags (ELF-spezifisch!)
MULTIBOOT_HEADER_FLAGS	equ	MULTIBOOT_PAGE_ALIGN | MULTIBOOT_MEMORY_INFO
MULTIBOOT_HEADER_CHKSUM	equ	-(MULTIBOOT_HEADER_MAGIC + MULTIBOOT_HEADER_FLAGS)
MULTIBOOT_EAX_MAGIC	equ	0x2badb002

SETUPSEG		equ	0x9000

;
;   System
;

[GLOBAL startup]
[GLOBAL startup_ap]
[GLOBAL idt_desc_global]
[GLOBAL __builtin_delete]
[GLOBAL __pure_virtual]
[GLOBAL __cxa_pure_virtual]
[GLOBAL _ZdlPv]
[GLOBAL __cxa_atexit]
[GLOBAL gdt_desc_global]

[EXTERN CPUstartup]
[EXTERN guardian]
[EXTERN ap_stack]
[EXTERN _init]
[EXTERN _fini]
[EXTERN __VESA_DETECTION_CODE_START__]
[EXTERN __VESA_DETECTION_CODE_END__]

[SECTION .text]

; Startup-Code fuer den BSP
startup:
	jmp skip_multiboot_hdr

multiboot_header:
	align 4
	dd MULTIBOOT_HEADER_MAGIC
	dd MULTIBOOT_HEADER_FLAGS
	dd MULTIBOOT_HEADER_CHKSUM

skip_multiboot_hdr:
; GCC-kompilierter Code erwartet das so.
	cld

	cmp eax,MULTIBOOT_EAX_MAGIC
;	jne floppy_boot
;
; GDT setzen (notwendig, falls wir durch GRUB geladen wurden)
;
	lgdt	[gdt_desc_global]
	jmp floppy_boot
vesa_detection:
	;VESA Detection Code ins vormalige SETUPSEG Segment relozieren
	mov ecx, __VESA_DETECTION_CODE_END__
	sub ecx, __VESA_DETECTION_CODE_START__
	mov esi, __VESA_DETECTION_CODE_START__	
	mov edi, SETUPSEG*0x10
	cld
	rep movsd
	jmp 0x20:0

resume_from_vesa:
	mov	ax,0x10
	mov	ds,ax
	lgdt	[gdt_desc_global]
floppy_boot:

; Globales Datensegment

	mov	ax,0x10
	mov	ds,ax
	mov	es,ax
	mov	fs,ax
	mov	gs,ax

; Stack festlegen

	mov	ss,ax
; cs Segment Register neu laden
; notwendig, falls wir über einen mulitboot specification bootloader geladen
; werden. Beim Start über Diskette wäre es nicht notwendig, tut aber auch nicht
; weh.
	jmp 0x8:load_cs
load_cs:
	mov	esp,init_stack+4096

; Unterbrechungsbehandlung sicherstellen

	call	setup_idt
	call	reprogram_pics

; Aufruf des C-Codes

	call	_init		; Konstruktoren globaler Objekte ausfuehren
	push	$1			; isBSP = 1
	call	CPUstartup	; C/C++ Level System
	pop		eax
	call	_fini		; Destruktoren
	hlt

; Startup-Code fuer die APs
startup_ap:

; GCC-kompilierter Code erwartet das so.
	cld

; richtige gdt initalisieren
	mov	ax,0x10
	mov	ds,ax
	lgdt	[gdt_desc_global]

; Restliche Segmentdeskriptoren laden

	mov	es,ax
	mov	fs,ax
	mov	gs,ax

; Stack festlegen, dieser wird ueber globale Variable uebergeben (ap_stack)

	mov	ss,ax
	mov	esp,[ap_stack]
;
; Unterbrechungsbehandlung sicherstellen
;
	lidt	[idt_desc_global]

; Aufruf des C-Codes
	push	$0			; isBSP = 0
	call	CPUstartup	; C/C++ Level System
	pop 	eax
	hlt

; Default Interrupt Behandlung

; Spezifischer Kopf der Unterbrechungsbehandlungsroutinen

%macro wrapper 1
wrapper_%1:
	push	eax
	mov	al,%1
	jmp	wrapper_body
%endmacro

; ... wird automatisch erzeugt.
%assign i 0
%rep 256
wrapper i
%assign i i+1
%endrep

; Gemeinsamer Rumpf
wrapper_body:
	cld			; das erwartet der gcc so.
	push	ecx		; Sichern der fluechtigen Register
	push	edx
	and	eax,0xff	; Der generierte Wrapper liefert nur 8 Bits
	push	eax		; Nummer der Unterbrechung uebergeben
	call	guardian
	add	esp,4		; Parameter vom Stack entfernen
	pop	edx		; fluechtige Register wieder herstellen
	pop	ecx
	pop	eax
	iret			; fertig!

;
; setup_idt
;
; Relokation der Eintraege in der IDT und Setzen des IDTR

setup_idt:
	mov	eax,wrapper_0	; ax: niederwertige 16 Bit
	mov	ebx,eax
	shr	ebx,16		; bx: hoeherwertige 16 Bit
	mov	ecx,255		; Zaehler
.loop:	add	[idt+8*ecx+0],ax
	adc	[idt+8*ecx+6],bx
	dec	ecx
	jge	.loop

	lidt	[idt_desc_global]
	ret

;
; reprogram_pics
;
; Neuprogrammierung der PICs (Programmierbare Interrupt-Controller), damit
; alle 15 Hardware-Interrupts nacheinander in der idt liegen.

reprogram_pics:
	mov	al,0x11   ; ICW1: 8086 Modus mit ICW4
	out	0x20,al
	call	delay
	out	0xa0,al
	call	delay
	mov	al,0x20   ; ICW2 Master: IRQ # Offset (32)
	out	0x21,al
	call	delay
	mov	al,0x28   ; ICW2 Slave: IRQ # Offset (40)
	out	0xa1,al
	call	delay
	mov	al,0x04   ; ICW3 Master: Slaves an IRQs
	out	0x21,al
	call	delay
	mov	al,0x02   ; ICW3 Slave: Verbunden mit IRQ2 des Masters
	out	0xa1,al
	call	delay
	mov	al,0x03   ; ICW4: 8086 Modus und automatischer EIO
	out	0x21,al
	call	delay
	out	0xa1,al
	call	delay

	mov	al,0xff		; Hardware-Interrupts durch PICs
	out	0xa1,al		; ausmaskieren. Nur der Interrupt 2,
	call	delay		; der der Kaskadierung der beiden
	mov	al,0xfb		; PICs dient, ist erlaubt.
	out	0x21,al

	ret

; delay
;
; Kurze Verzoegerung fuer in/out Befehle.

delay:
	jmp	.L2
.L2:	ret

; Die Funktion wird beim abarbeiten der globalen Konstruktoren aufgerufen
; (unter Linux). Das Label muss definiert sein (fuer den Linker). Die
; Funktion selbst kann aber leer sein, da bei StuBs keine Freigabe des
; Speichers erfolgen muss.

__pure_virtual:
__cxa_pure_virtual:
__builtin_delete:
_ZdlPv:
__cxa_atexit:
        ret

[SECTION .data]

;  'interrupt descriptor table' mit 256 Eintraegen.

align	4 
idt:

%macro idt_entry 1
	dw	(wrapper_%1 - wrapper_0) & 0xffff
	dw	0x0008
	dw	0x8e00
	dw	((wrapper_%1 - wrapper_0) & 0xffff0000) >> 16
%endmacro

; ... wird automatisch erzeugt.

%assign i 0
%rep 256
idt_entry i
%assign i i+1
%endrep

idt_desc_global:
	dw	256*8-1		; idt enthaelt 256 Eintraege
	dd	idt

;   Stack und interrupt descriptor table im BSS Bereich

[SECTION .bss]

init_stack:
	resb	4096

[SECTION .data]
;
; Descriptor-Tabellen

   
align	4 
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
	
	dw	0xFFFF		; 64k Groesse
	dw	0x0000		; base address=0x90000 (SETUPSEG)
	dw	0x9209		; data read/write
	dw	0x0000		; byte granularity
	
	dw	0xFFFF		; 64k Groesse
	dw	0x0000		; base address=0x90000 (SETUPSEG)
	dw	0x9A09		; code read/exec
	dw	0x0000		; byte granularity

gdt_desc_global:
	dw	0x28		; GDT Limit=24, 4 GDT Eintraege
	dd	gdt		; Physikalische Adresse der GDT
