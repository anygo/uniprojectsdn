; $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
; kate: encoding ISO-8859-15;
; vim: set fileencoding=latin-9:
; -*- mode: c; coding: latin-9 -*-

%include "machine/toc.inc"

; EXPORTIERTE FUNKTIONEN

[GLOBAL toc_switch]
[GLOBAL toc_go]

; IMPLEMENTIERUNG DER FUNKTIONEN

[SECTION .text]

; TOC_GO : Startet den ersten Prozess ueberhaupt.
;
; C Prototyp: void toc_go (struct toc* regs);

toc_go:
	mov eax, [4 + esp]
	mov ebx, [ebx_offset + eax]
	mov esi, [esi_offset + eax]
	mov edi, [edi_offset + eax]
	mov ebp, [ebp_offset + eax]
	mov esp, [esp_offset + eax]

	ret



; TOC_SWITCH : Prozessumschaltung. Der aktuelle Registersatz wird     
;              gesichert und der Registersatz des neuen "thread of control"
;              wird in den Prozessor eingelesen.  
;
; C Prototyp: void toc_switch (struct toc* regs_now,
;                              struct toc* reg_then);

toc_switch:
	mov eax, [4 + esp]
	mov [eax + ebx_offset], ebx
	mov [eax + esi_offset], esi
	mov [eax + edi_offset], edi
	mov [eax + ebp_offset], ebp
	mov [eax + esp_offset], esp

	mov eax, [8 + esp]
	mov ebx, [ebx_offset + eax]
	mov esi, [esi_offset + eax]
	mov edi, [edi_offset + eax]
	mov ebp, [ebp_offset + eax]
	mov esp, [esp_offset + eax]

	ret


