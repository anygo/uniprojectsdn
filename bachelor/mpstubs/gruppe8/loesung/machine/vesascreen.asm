

[GLOBAL vesa_init_pmstub]
[GLOBAL vesa_get_mode_info_pmstub]
[GLOBAL vesa_set_mode_pmstub]

[section .text]

;Stub, um Vesafunktionen im Realmode aufzurufen
; 

vesa_init_pmstub:
    cli
    push ebp
    push ebx
    push esi
    push edi
    push 0    ;Leerer zweiter Parameter
    push 0    ;Fuehre VESA OP 0 Aus
    call 0x20:0x0
    add esp, 8
    pop edi
    pop esi
    pop ebx
    pop ebp
    and eax, 0xffff
    sti
    ret

;Hole informationen über jeweiligen Modus
;
;C-Prototyp: int vesa_get_mode_info_pmstub(int mode_number);
; 

vesa_get_mode_info_pmstub:    
    mov eax,[esp+4]
    cli
    push ebp
    push ebx
    push esi
    push edi
    push eax  ;Modus, der abgefragt werden soll
    push 1    ;Fuehre VESA OP 1 aus
    call 0x20:0x0
    add esp, 8
    pop edi
    pop esi
    pop ebx
    pop ebp
    and eax, 0xffff
    sti
    ret

; Setze Modus mit der Nummer modenumber
;
;C-Prototyp int vesa_set_mode_pmstub(int mode_number);
;

vesa_set_mode_pmstub:
    mov eax,[esp+4]
    cli
    push ebp
    push ebx
    push esi
    push edi
    push eax  ;Modus, der gestetzt werden soll
    push 2    ;Fuehre VESA OP 2 aus
    call 0x20:0x0
    add esp, 8
    pop edi
    pop esi
    pop ebx
    pop ebp
    and eax, 0xffff
    sti
    ret

