// $Date: 2009-08-14 16:19:58 +0200 (Fri, 14 Aug 2009) $, $Revision: 2209 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#ifndef __MP_REGISTERS_H_
#define __MP_REGISTERS_H_

/// Static Assertion
#define ct_assert(e) extern char (*ct_assert(void)) [sizeof(char[1 - 2*!(e)])]

/// Intel MP Spec, p. 4-3
#define MPFPS_ID_SIGNATURE      (('_'<<24)|('P'<<16)|('M'<<8)|'_')

/// Intel MP Spec, p. 4-4
#define COMPATIBILITY_PIC       (1 << 7)
/// \copydoc COMPATIBILITY_PIC
#define COMPATIBILITY_VIRTWIRE      0x0

/// Intel MP Spec, p. 4-7
#define MPCT_PROCESSOR          0x0
/// \copydoc MPCT_PROCESSOR
#define MPCT_BUS                0x1
/// \copydoc MPCT_PROCESSOR
#define MPCT_IOAPIC             0x2
/// \copydoc MPCT_PROCESSOR
#define MPCT_IOINT              0x3
/// \copydoc MPCT_PROCESSOR
#define MPCT_LINT               0x4

/// Intel MP Spec, p. 4-8
#define CPU_ENABLED             0x1
/// \copydoc CPU_ENABLED
#define CPU_BOOTPROCESSOR       0x2

/*! \brief MP floating pointer structure
 * 
 *  Intel MP Spec, p. 4-3
 */
struct mpfps
{
    char signature[4]; ///< signature "_MP_"
    unsigned int physptr; ///< physical address pointer (MP config table address)
    unsigned char length; ///< length of the structure in 16-byte units
    unsigned char spec_rev; ///< MP version (0x01 = 1.1; 0x04 = 1.4)
    unsigned char checksum; ///< checksum (overall sum must be 0)
    unsigned char feature1; ///< =0: there is an MP config table; otherwise: default config
    unsigned char feature2; ///< bit 7 set: IMCR and PIC mode; otherwise: virtual wire mode
    unsigned char feature3; ///< reserved (0)
    unsigned char feature4; ///< reserved (0)
    unsigned char feature5; ///< reserved (0)
};
ct_assert(sizeof(struct mpfps) == 4*4);

/*! \brief MP config table header
 * 
 *  Intel MP Spec, p. 4-5
 */
struct mpcth
{
    char signature[4]; ///< signature"PCMP"
    unsigned short length; ///< table length in bytes
    char spec_rev;
    char checksum;
    char oemid[8]; ///< system manufacturer ID
    char productid[12]; ///< product family ID
    unsigned int oemptr; ///< pointer to an OEM config table (0 if non-existent)
    unsigned short oemsize; ///< length of the OEM config table in bytes
    unsigned short count; ///< number of entries in the base table
    unsigned int lapic; ///< memory-mapped address of local APIC
    unsigned short exttbllen; ///< length of the extended entries in bytes
    char exttblchksum;
    char reserved; ///< reserved (0)
};
ct_assert(sizeof(struct mpcth) == 4*11);

/*! \brief Processor Entry
 * 
 *  Intel MP Spec, p. 4-7
 */
struct mpct_processor
{
    unsigned char type; // type = 0
    unsigned char lapicid;
    unsigned char lapicver;
    unsigned char cpuflag; // bit 0 (EN): enable/disable; bit 1 (BP): BSP/AP
    unsigned int cpusignature;
    unsigned int featureflags;
    unsigned int reserved[2];
};
ct_assert(sizeof(struct mpct_processor) == 4*5);

/*! \brief Bus Entry
 *  Intel MP Spec, p. 4-10
 */
struct mpct_bus
{
    unsigned char type; // type = 1
    unsigned char busid;
    unsigned char bustype[6];
};
ct_assert(sizeof(struct mpct_bus) == 4*2);

/*! \brief IOAPIC Entry
 *  
 *  Intel MP Spec, p. 4-12
 */
struct mpct_ioapic
{
    unsigned char type; // type = 2
    unsigned char apicid;
    unsigned char apicver;
    unsigned char flags;
    unsigned int apicaddr;
};
ct_assert(sizeof(struct mpct_ioapic) == 4*2);

/*! \brief  I/O Interrupt Entry
 *  
 *  Intel MP Spec, p. 4-13
 */
struct mpct_int
{
    unsigned char type; // type = 3/4
    unsigned char irqtype;
    unsigned short irqflag;
    unsigned char srcbus;
    unsigned char srcbusirq;
    unsigned char dstapic;
    unsigned char dstirq;
};
ct_assert(sizeof(struct mpct_int) == 4*2);

#endif /* MP_REGISTERS_H_ */
