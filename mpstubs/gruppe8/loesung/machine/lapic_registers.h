// $Date: 2009-09-17 12:48:38 +0200 (Thu, 17 Sep 2009) $, $Revision: 2220 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

/*! \file 
 *  \brief Strukturen und Makros zum Zugriff auf den Local APIC
 */

#ifndef __LAPIC_REGISTERS_H__
#define __LAPIC_REGISTERS_H__

/*! \brief Basisadresse der local APIC-Register
 *  
 *  System Programming Guide 3A, p. 9-8
 */
#define LAPIC_BASE 0xfee00000

/*! \brief Einstellungsmöglichkeiten und Modi für die ICR_H und ICR_L Register
 *  
 *  System Programming Guide 3A, p. 9-39
 */
#define DESTINATION_SHORTHAND_NO			0x0
/// \copydoc DESTINATION_SHORTHAND_NO
#define DESTINATION_SHORTHAND_SELF			0x1
/// \copydoc DESTINATION_SHORTHAND_NO
#define DESTINATION_SHORTHAND_ALLINCSELF	0x2
/// \copydoc DESTINATION_SHORTHAND_NO
#define DESTINATION_SHORTHAND_ALLEXCSELF	0x3

/// \copydoc DESTINATION_SHORTHAND_NO
#define TRIGGER_MODE_EDGE					0x0
/// \copydoc TRIGGER_MODE_EDGE
#define TRIGGER_MODE_LEVEL					0x1

/// \copydoc DESTINATION_SHORTHAND_NO
#define LEVEL_DEASSERT						0x0
/// \copydoc LEVEL_DEASSERT
#define LEVEL_ASSERT						0x1

/// \copydoc DESTINATION_SHORTHAND_NO
#define DELIVERY_STATUS_IDLE				0x0
/// \copydoc DESTINATION_SHORTHAND_NO
#define DELIVERY_STATUS_SEND_PENDING		0x1

/// \copydoc DESTINATION_SHORTHAND_NO
#define DESTINATION_MODE_PHYSICAL			0x0
/// \copydoc DESTINATION_SHORTHAND_NO
#define DESTINATION_MODE_LOGICAL			0x1

/// \copydoc DESTINATION_SHORTHAND_NO
#define DELIVERY_MODE_FIXED					0x0
/// \copydoc DESTINATION_SHORTHAND_NO
#define DELIVERY_MODE_LOWESTPRI				0x1
/// \copydoc DESTINATION_SHORTHAND_NO
#define DELIVERY_MODE_SMI					0x2
// Reserved			 						0x3
/// \copydoc DESTINATION_SHORTHAND_NO
#define DELIVERY_MODE_NMI					0x4
/// \copydoc DESTINATION_SHORTHAND_NO
#define DELIVERY_MODE_INIT					0x5
/// \copydoc DESTINATION_SHORTHAND_NO
#define DELIVERY_MODE_STARTUP				0x6
// Reserved									0x7

/*! \brief Einstellungsmöglichkeiten und Modi für das DFR Register 
 *  
 *  System Programming Guide 3A, p. 9-48
 */
#define DESTINATION_MODEL_CLUSTER			0x0
/// \copydoc DESTINATION_MODEL_CLUSTER
#define DESTINATION_MODEL_FLAT				0xf

/*! \brief Einstellungsmöglichkeiten und Modi für das SVR Register
 *  
 *  System Programming Guide 3A, p. 9-64
 */
#define FOCUS_CPU_ENABLED					0x0
/// \copydoc FOCUS_CPU_ENABLED
#define FOCUS_CPU_DISABLED					0x1

/// \copydoc FOCUS_CPU_ENABLED
#define LAPIC_DISABLED						0x0
/// \copydoc FOCUS_CPU_ENABLED
#define LAPIC_ENABLED						0x1

/*! \brief Local APICID Register für P6 und Pentium 
 * 
 *  siehe: System Programming Guide 3A, p. 9-13
 *  
 */
struct LAPICID_P // Pentium CPUs
{
	int	reserved_1:24,
		apic_id:4, ///< APIC ID
		reserved_2:4;
} __attribute__((packed));

/*! \brief Local APIC ID Register für Pentium IV und spätere
 *  
 *  siehe: System Programming Guide 3A, p. 9-13
 */
struct LAPICID_P4 // Pentium 4 and Xeon CPUs
{
	int	reserved_1:24,
		apic_id:8; ///< APIC ID
} __attribute__((packed));

/*! \brief Local APIC Version Register
 *  
 *  siehe: System Programming Guide 3A, p. 9-15
 */
struct LAPICVER
{
	int	version:8, ///< Version (0x14 for P4s and Xeons)
		reserved_1:8,
		max_lvt_entry:8, ///< Maximum LVT Entry
		reserved_2:8;
} __attribute__((packed));

/*! \brief Interrupt Command Register Low
 *  
 *  siehe: System Programming Guide 3A, p. 9-39
 */
struct ICR_L
{
	int	vector:8, ///< Vector
		delivery_mode:3, ///< Delivery Mode
		destination_mode:1, ///< Destination Mode
		delivery_status:1, ///< Delivery Status
		reserved_1:1,
		level:1, ///< Level
		trigger_mode:1, ///< Trigger Mode
		reserved_2:2,
		destination_shorthand:2, ///< Destination Shorthand
		reserved_3:12;
} __attribute__((packed));

/*! \brief Interrupt Command Register High
 *  
 *  siehe: System Programming Guide 3A, p. 9-39
 */
struct ICR_H
{
	int	reserved:24,
		destination_field:8; ///< Destination Field
} __attribute__((packed));

/*! \brief Logical Destination Register
 * 
 *  siehe: System Programming Guide 3A, p. 9-47
 */
struct LDR
{
	int reserved:24,
		lapic_id:8; ///< Logical APIC ID
} __attribute__((packed));

/*! \brief Destination Format Register
 * 
 *  System Programming Guide 3A, p. 9-48
 */
struct DFR
{
	int reserved:28,
		model:4; // Model (Flat vs. Cluster)
} __attribute__((packed));

/*! \brief Task Priority Register
 * 
 *  System Programming Guide 3A, p. 9-58
 */
struct TPR
{
	int task_prio_sub:4, ///< Task Priority Sub-Class
		task_prio:4, ///< Task Priority
		reserved:24;
} __attribute__((packed));

/*! \brief Spurious Interrupt Vector Register
 * 
 *  System Programming Guide 3A, p. 9-64
 */ 
struct SVR
{
	int spurious_vector:8, ///< Spurious Vector
		apic_enable:1, ///< APIC Software Enable/Disable
		focus_processor_checking:1, ///< Focus Processor Checking
		reserved:22;
} __attribute__((packed));

/// Union über alle LAPIC-Register, um die Register generisch verwenden zu können
union LAPICRegister {
	struct LAPICID_P		lapicid_p; 
	struct LAPICID_P4		lapicid_p4;
	struct LAPICVER			lapicver;
	struct ICR_L			icr_l;
	struct ICR_H			icr_h;
	struct LDR				ldr;
	struct DFR				dfr;
	struct TPR				tpr;
	struct SVR			    svr;
	unsigned int value;
} __attribute__((packed));

typedef union LAPICRegister LAPICRegister_t;

#endif
