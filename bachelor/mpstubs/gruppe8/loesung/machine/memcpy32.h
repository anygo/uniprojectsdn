#ifndef MEMCPY32_H
#define MEMCPY32_H

/*****************************************************************************
 * Betriebssysteme                                                           *
 *---------------------------------------------------------------------------*
 *                                                                           *
 *                                  MEMCPY32                                 *
 *                                                                           *
 *---------------------------------------------------------------------------*
 * Nachdem Zeichenweisses Kopieren in ner for-schleife was fuer Weicheier    *
 * ist, kommt hier das memcopy fuer echte Maenner: Wir benutzen soweit       *
 * moeglich 32 bit transfers, und die schleife ueberlassen wir auch dem      *
 * Prozessor, mittels 'rep' prefix. HOHOHO, MEHR POWER!                      *
 *****************************************************************************/

/* BENUTZTE FUNKTIONEN */

#ifdef __cplusplus
extern "C" {
#endif
  
  void memcpy32(unsigned long n, void * src, void * dst);
  void memset32(unsigned long n, void * dst, unsigned long fillwith);

#ifdef __cplusplus
};
#endif

#endif
// MEMCPY32_H
