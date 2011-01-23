;*****************************************************************************
;* Betriebssysteme                                                           *
;*---------------------------------------------------------------------------*
;*                                                                           *
;*                                  MEMCPY32                                 *
;*                                                                           *
;*---------------------------------------------------------------------------*
;* Nachdem Zeichenweisses Kopieren in ner for-schleife was fuer Weicheier    *
;* ist, kommt hier das memcopy fuer echte Maenner: Wir benutzen soweit       *
;* moeglich 32 bit transfers, und die schleife ueberlassen wir auch dem      *
;* Prozessor, mittels 'rep' prefix. HOHOHO, MEHR POWER!                      *
;*****************************************************************************

; EXPORTIERTE FUNKTIONEN

[GLOBAL memcpy32]
[GLOBAL memset32]

; IMPLEMENTIERUNG DER FUNKTIONEN

[SECTION .text]
  
; memcpy32: Kopieren von Speicherbereichen, mit uuuuuuuunglaublicher
; Geschwindigkeit.
;  C: void memcpy32(unsigned long n, void * src, void * dst);

memcpy32:
  push  ebp
  mov ebp,esp
  push esi
  push edi
  mov esi,[12+ebp]
  mov edi,[16+ebp]
  cld
  mov ecx,[8+ebp]
  shr    ecx,1      ; Ungerade Anzahl Bytes? }
  jnc    memcpy2    ; Gerade Anzahl Bytes ==> zu memcpy2 springen
  movsb             ; Ein Byte kopieren
memcpy2:
  shr    ecx,1      ; Ungerade Anzahl Words?
  jnc    memcpy1    ; Gerade Anzahl Words ==> zu memcpy1 springen
  movsw             ; Ein Word kopieren
memcpy1:
  rep movsd         ; Soviele DWords kopieren wie in ecx angegeben
  pop edi
  pop esi
  pop ebp
  ret
  
; memcpy32: Kopieren von Speicherbereichen, mit uuuuuuuunglaublicher
; Geschwindigkeit.
;
;  C: void memset32(unsigned long n, void * dst, unsigned long fillwith);
;  fillwith ist natuerlich womit gefuellt werden soll.
; Vorsicht: schlecht vorhersagbares Ergebnis wenn n nicht durch 4 (=wortlaenge)
; teilbar ist!
memset32:
  push  ebp
  mov ebp,esp
  push edi
  mov edi,[12+ebp]
  mov ecx,[8+ebp]
  mov eax,[16+ebp]
  cld
  shr    ecx,1      ; Ungerade Anzahl Bytes? }
  jnc    memset2    ; Gerade Anzahl Bytes ==> zu copymem2 springen
  stosb             ; Ein Byte schreiben (al)
memset2:
  shr    ecx,1      ; Ungerade Anzahl Words?
  jnc    memset1    ; Gerade Anzahl Words ==> zu copymem1 springen
  stosw             ; Ein Word schreiben (ax)
memset1:
  rep stosd         ; Soviele DWords schreiben wie in ecx angegeben...
  pop edi           ;  ... mit dem wert aus eax
  pop ebp
  ret
