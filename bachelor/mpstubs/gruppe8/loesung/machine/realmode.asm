;Umschalten in den Real-Mode und wieder zurück

SETUPSEG	equ 0x9000	    ; Setupsegment wiederverwenden

RM_SEG_DESC	equ 0x18

[EXTERN idt_desc_global]
[EXTERN gdt_desc_global]
[GLOBAL rm_start]
[GLOBAL VBEInfoBlock]
[GLOBAL ModeInfoBlock]

[SECTION .vesa_detection]
[BITS 16]

rm_start:
    ;save which operation to do in rm
    mov eax, [esp+8]
    mov [dword (SETUPSEG*0x10)+(vesa_op-rm_start)], eax
    mov eax, [esp+12]
    mov [dword (SETUPSEG*0x10)+(param-rm_start)], eax

    mov eax, RM_SEG_DESC
    mov ds, eax
    mov	ss, eax
    mov es, eax
    mov fs, eax
    mov gs, eax
    ;save source-esp
    mov [source_esp-rm_start], esp
    mov eax, cr0
    and al, 0xfe
    mov cr0, eax
    jmp word 0x9000:(real_mode-rm_start)

real_mode:
    mov ax, SETUPSEG
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax
    mov sp, 0xfffe
    lidt [idt_real-rm_start]
; Auszuführende VESA-Funktion auswählen 
    cmp dword[vesa_op-rm_start], 0 
    je vesa_init
    cmp dword[vesa_op-rm_start], 1 
    je vesa_get_mode_info
    cmp dword[vesa_op-rm_start], 2 
    je vesa_set_mode

vesa_init:
    mov di, VBEInfoBlock-rm_start
    mov ax, 0x4f00
    int 0x10
    cmp ax, 0x004f
    jne init_fail
    push 1
    jmp get_back
init_fail:
    push 0
    jmp get_back

vesa_get_mode_info:
    mov ecx, [param-rm_start] ; Modus als Parameter
    mov ax, 0x4f01	      ; Befehlscode
    mov di, ModeInfoBlock-rm_start
    int	0x10
    cmp ax, 0x004f 
    jne get_mode_fail
    push 1
    jmp get_back
get_mode_fail:
    push 0
    jmp get_back

vesa_set_mode:
    mov ebx, [param-rm_start]
    mov ax, 0x4f02 
    int 0x10
    cmp ax, 0x004f
    jne set_mode_fail
    push 1
    jmp get_back
set_mode_fail:
    push 0
    jmp get_back


    jmp vesa_pmode

    mov ax, 0x0013
    int 0x10
    mov ax, 0xa000
    mov es, ax
    mov byte [es:0], 45
    mov byte [es:1], 0xff 
    mov byte [es:2], 0xf0 
    mov byte [es:3], 42
    mov ax, 0x0003
    int 0x10
    jmp vbe_fail
    jmp wait

vesa_pmode:
    mov ax, 0x4f0a 
    mov bx, 0
    int 0x10
    cmp al, 0x4f
    jne vbe_fail
    shr ax, 8
    cmp al, 0x0
    jne vbe_fail
    mov byte [4], 's'
    mov byte [5], 7
    jmp vbe_succ
vbe_fail:
    mov byte [4], 'f'
    mov byte [5], 7
    add al, 0x30
    mov byte [6], al
    mov byte [7], 7
vbe_succ:

;    jmp get_back

wait:
    jmp wait

get_back:
    pop bx
    mov ax, SETUPSEG
    mov ds, ax
    lidt[idt_tmp-rm_start]	;temporäre idt & gdt für den Wechsel in den PM
    lgdt[gdt_tmp-rm_start]  ;wirkliche gdt & idt sind aus dem RM nicht zu erreichen
    mov eax,cr0
    or eax,1
    mov cr0,eax
    jmp dword 0x08:(SETUPSEG*0x10)+(pmode-rm_start)

[BITS 32]

pmode:
    mov eax, 0x10
    mov ds, eax
    lgdt [gdt_desc_global]   ;stelle die globale idt/gdt wieder her
    lidt [idt_desc_global]
    mov es, eax		     ;Segmentregister wieder 'richtig' initialisieren
    mov fs, eax
    mov gs, eax
    mov ss, eax
    mov esp, [SETUPSEG*0x10+(source_esp-rm_start)]   ;ursprünglicher Stack des Aufrufers
    mov ax, bx
    retf

source_esp:
    dd		0
vesa_op:
    dd		0
param:
    dd		0
idt_real: 
    dw		0x3ff
    dd		0

idt_tmp:
	dw	0		; idt limit=0
	dw	0,0		; idt base=0L

align 4
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

gdt_tmp:
	dw	0x18		; GDT Limit=24, 3 GDT Eintraege
	dd     SETUPSEG*0x10+(gdt-rm_start); Physikalische Adresse der GDT

VBEInfoBlock:
VBESignature:
	db	'VBE2'
VBEVersion:
	dw	0
OEMStringPtr:
	dd	0
Capabilities:
	db	0,0,0,0
VideoModePtr:
	dd	0
TotalMemory:
	dw	0
OEMSoftwareRev:
	dw	0
OEMVendorNamePtr:
	dd	0
OEMProductNamePtr:
	dd	0
OEMProductRevPtr:
	dd	0
Reserved:
	TIMES 222 db 0
OEMData:
	TIMES 512 db 0

ModeInfoBlock:
	TIMES 256 db 0


VBE_PM_INTERFACE:
VBE_STATUS:
    dw 0	    ; VBE Function return status
VBE_SEG:
    dw 0	    ; Real-Mode Segment of VBE PM Structure
VBE_OFF:
    dw 0	    ; Offset of VBE PM Structure
VBE_LEN:
    dw 0	    ; Length of VBE PM Structure
