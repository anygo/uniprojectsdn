// $Date: 2009-09-22 15:20:27 +0200 (Tue, 22 Sep 2009) $, $Revision: 2221 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#include "machine/apicsystem.h"
#include "device/cgastr.h"
#include "machine/io_port.h"
#include "machine/lapic.h"
#include "object/debug.h"
#include "machine/acpi.h"

// global object definition
APICSystem system;

void APICSystem::detectSystemType() {
    // System base memory
    unsigned short memsize = *((unsigned short *) (0x413));

    //DBG << "memsize is " << dec << memsize << endl;

    /*
     * The MPFP structure can be located in one of three different memory areas:
     * 1) In the upper 1K of the EBDA (BIOS data area 0x40:0e holds segment of EBDA)
     * 2) In the upper 1K of the base RAM (639K-640K or 511K-512K)
     * 3) In the BIOS ROM (0x0f0000-0x0fffff) 
     */

    if (searchConfig(*((unsigned short *) 0x40e) << 4, 1024) || 
            searchConfig(memsize * 1024, 1024) || 
            searchConfig(511 * 1024, 1024) || 
            searchConfig(639 * 1024, 1024) || 
            searchConfig(0x000f0000, 0x000fffff - 0x000f0000 + 1)) {
        system_type = MP_APIC;
        return;
    }

    /*
     * We may have processors with hyperthreading installed,
     * which will only be recognized using ACPI. 
     * As the MPS based detection may already have detected
     * 1 CPU, we have to set this back to 0
     */

    numberOfCPUs = 0;

    system_type = queryACPI();
    if (system_type != MP_APIC) {
        //DBG << "SMP Detection using ACPI: FAIL" << endl;
        //DBG << "Did not detect SMP system." << endl;
        numberOfCPUs = 1;
        onlineCPUs = 1;
    }
    return;
}

APICSystem::SystemType APICSystem::queryACPI() {
    using namespace ACPI_Subsystem;

    RSDP *rsdp = 0; //Root System Description Pointer
    RSDT *rsdt = 0; //Root System Description Table
    //XSDT *xsdt = 0; //Xtended System Description Table
    MADT *madt = 0; //Multiple APIC Description Table
    int rsdt_num_entries; //Number of SDT pointers in RSDT
    //int xsdt_num_entries; //Number of SDT pointers in XSDT
    int idx;

    /* 
     * look for the root system description pointer
     */
    if ( !( rsdp = RSDP::find() ) ) {
        //DBG << "No ACPI..." << endl;
        return UNDETECTED;
    }

    /*	 
     * look for the root system description table
     */
    if ( !( rsdt = rsdp->getRSDTPtr() ) ) {
        //DBG << "No ACPI Root System Description Table :-(" << endl;
    }
	
	/*
	 * parse the root system description table and search for the MADT
	 */
	rsdt_num_entries = rsdt->getNumSDTPtrs();
	//DBG << "RSDT contains " << rsdt_num_entries << " SDT pointers" << endl;

	for ( idx = 0; idx < rsdt_num_entries; idx++) {
	    System_Description_Table *entry = rsdt->getSDTPtr( idx );

	    //DBG << "SDT pointer" << idx << ": type ";
	    //DBG << entry->getSignature()[0] << entry->getSignature()[1];
	    //DBG << entry->getSignature()[2] << entry->getSignature()[3];
	    //DBG << endl;

	    if ( entry->getType() == System_Description_Table::MADT ) {
	        //DBG << "MADT found (yay) !!!" << endl;
	        madt = (MADT *) entry;
	    }	
	}

	/*
	 * no MADT found -> give up 
	 */
	if ( !madt ) {
	    //		DBG << "No MADT Ptr found in RSDT" << endl;

	    if ( rsdp->is_ACPI_1_0()) {
	        //DBG << "   and no XSDT given (ACPI 1.0)" << endl;
	        //DBG << "   -> giving up" << endl;
	    } else {
	        //DBG << "   and XSDT (ACPI>1.0) not supported yet";
	        //DBG << endl;
	        //DBG << "   PLEASE TELL THE BS(B) STAFF!" << endl;
	    }
	    return UNDETECTED;
	}

	/*
	 * parse the madt
	 */
	if ( madt->has_PCAT_COMPAT() )
		disablePICs();

	{
	APIC_Structure *entry = madt->getFirstAPICStructurePtr();
	APIC_Structure *end   = madt->getRearPtr();
	bool found_bsp = false;
	idx=0;

//	DBG << "MADT begins at" << (void *) madt << endl;
//	DBG << "APIC Structures begin at" << (void *) entry << endl;
//	DBG << "MADT ends at" << (void *) end << endl;

	while ( entry < end ) {	
		int entry_type = entry->getType();
	
//		DBG << "APIC Structure " << idx++ << " at " << (void *) entry;
//		DBG << " type " << entry_type << endl;

		if ( entry->getLength() == 0 ) {
//			DBG << "found zero length APIC Structure in MADT... WTF?" << endl;
			break;
		}

		switch ( entry_type ) { 
		/*
		 * found a LAPIC
		 */
		case APIC_Structure::LOCAL_APIC:
		{	
		    Local_APIC_Structure *lapic =
		            (Local_APIC_Structure *) entry;
		    unsigned char lapic_id;
		    unsigned char current_cpu;

		    if ( !lapic->isEnabled() )
		        break;

		    if ( numberOfCPUs == CPU_MAX )
		        break;

		    /* get the cpu id from the lapic instead
		     * of using our favourite one */
		    lapic_id = lapic->getAPIC_ID();
		    current_cpu = lapic->getProcessorID();

		    //				DBG << "detected APIC with ID ";
		    //				DBG << (int) lapic_id;
		    //				DBG << " and corresponding CPU with ID ";
		    //				DBG << (int) current_cpu << endl;

		    if ( current_cpu >= CPU_MAX ) {
		        //					DBG << "detected CPU ID >= max supported #CPUs" << endl;
		        break;
		    }

		    ++numberOfCPUs;

		    lapicID[current_cpu] = lapic_id;
		    //unkown: lapicVersion,	cpuFamily, cpuModel

		    //according to the spec (5.2.11.4.1)
		    if ( !found_bsp ) {
		        bspID = lapic_id;
		        //as done in readMPCT and checked for in bootCPU
		        found_bsp = true;
		    }

		    break;
		}
		/* 
		 * found an IO APIC
		 */
		case APIC_Structure::IO_APIC:
		{	
		    IO_APIC_Structure *io_apic =
		            (IO_APIC_Structure *) entry;

		    ioapicID = io_apic->getAPIC_ID();
		    //Setting IOAPICID
		    unsigned int id = ioapicID << 4;
		    (*((volatile unsigned int*) 0xfec00000)) = 0;
		    (*((volatile unsigned int*) 0xfec00010)) = id;
		    //				DBG << "detected IO APIC with ID "
		    //				    << (int) ioapicID << endl;
		}
		}

		entry = entry->getNextAPICStructurePtr();
	}
	}
	
	if ( !rsdp->is_ACPI_1_0() )	{
	    //DBG << "WARNING: ACPI > 1.0 detected, but will not parse XSDT" << endl;
	}

	if (numberOfCPUs > 1) {
	    return MP_APIC;
	} else {
	    return UP_APIC;
	}
}

unsigned char APICSystem::getCPUID()
{
	if (system_type == UNDETECTED) {
		return 0;
	} else {
		return lapic.getCPUID();
	}
}

bool APICSystem::searchConfig(unsigned long base, unsigned long length) {
    struct mpfps *mpfps;

    //DBG << "searching MPFP structure starting at " << hex << base << " (length=" << dec << length << ")" << endl;

    // structure is 16 bytes long and beginning on a 16-byte boundary
    unsigned long *bp = (unsigned long *) (base & ~0x0000000f);

    for ( ; length > 0; bp += 4, length -= 16) {
        if (*bp != MPFPS_ID_SIGNATURE) {
            continue;
        }

        mpfps = (struct mpfps *) bp;

        // length reasonable?
        if (mpfps->length != 1) {
            //DBG << "invalid size (16*" << (int)mpfps->length << ")" << endl;
            continue;

            // checksum OK?
        } else if (!mpbChecksumOK((unsigned char *) bp, mpfps->length * sizeof(struct mpfps))) {
            //DBG << "invalid checksum" << endl;
            continue;

            // valid version of the MP spec?
        } else if (mpfps->spec_rev != 1 && mpfps->spec_rev != 4) {
            //DBG << "unknown MPFS spec_rev = " << hex << (int)mpfps->spec_rev << dec << endl;
            continue;
        }

        // extract compatibility mode from structure
        compatibilityMode = mpfps->feature2 & COMPATIBILITY_PIC;

        // switch to symmetric I/O if necessary
        if (compatibilityMode == COMPATIBILITY_PIC)
            disablePICs();

        // check for standard configuration
        if (mpfps->feature1 != 0) {
            // standard configuration
            //DBG << "Standard configuration detected." << endl;
        } else if (mpfps->physptr != 0x0) {
            // there is an MPCT
            //DBG << "MPCT detected." << endl;
            readMPCT((struct mpcth *) mpfps->physptr);
        }

        // the configuration that was found first is used
        /* In the P4 case, we should return false, in order
		   to give the ACPI SMP detection a chance */
        return (numberOfCPUs > 1);
    }

    return false; // no signature was found
}

bool APICSystem::mpbChecksumOK(unsigned char *start_address, int len) {
    int sum = 0;
    while (len--) {
        sum += *start_address++;
    }
    if ((sum & 0xff) == 0) {
        return true;
    } else {
        return false;
    }
}

void APICSystem::readMPCT(struct mpcth *mpcth) {
	int count = sizeof(*mpcth);
	unsigned char *mpt = ((unsigned char *) mpcth) + count;	// pointer to table

	// check signature "PCMP"
	if (((mpcth->signature)[0] != 'P') ||
		((mpcth->signature)[1] != 'C') ||
		((mpcth->signature)[2] != 'M') ||
		((mpcth->signature)[3] != 'P')) {
		// wrong signature
		return;
	}

	// check checksum
	if (! mpbChecksumOK((unsigned char *) mpcth, mpcth->length)) {
		// wrong checksum
		return ;
	}

	// investigate configuration blocks
	// loop while less bytes (count) have been read than the MPCT is long (length)
	while (count < mpcth->length) {
		switch (*mpt) {
		case MPCT_PROCESSOR:
			{
				struct mpct_processor *m = (struct mpct_processor *) mpt;
				if (m->cpuflag & CPU_ENABLED) {
					numberOfCPUs++;
					if (m->cpuflag & CPU_BOOTPROCESSOR) {
						// ID of the bootstrap CPU
						bspID = m->lapicid;
					}

					if (numberOfCPUs <= CPU_MAX) {
						int current = numberOfCPUs - 1;
						lapicID[current] = m->lapicid;
						lapicVersion[current] = m->lapicver;
						// save family and model
						cpuFamily[current] = (m->cpusignature & 0xf00) >> 8;
						cpuModel[current] = (m->cpusignature & 0xf0) >> 4;
					}
				}
				mpt += sizeof(*m); // set mpt to address after MPT_PROCESSOR structure
				count += sizeof(*m); // increase count by number of read bytes
				break;
			}
		case MPCT_IOAPIC:
			{
				struct mpct_ioapic *m = (struct mpct_ioapic *) mpt;
				mpt += sizeof(*m);
				count += sizeof(*m);
				// save I/O-APIC ID for later initialization
				ioapicID = m->apicid;
				break;
			}
		// the pieces of information from the other structures are not important for us
		case MPCT_BUS:
			{
				struct mpct_bus *m = (struct mpct_bus *) mpt;
				mpt += sizeof(*m);
				count += sizeof(*m);
				break;
			}
		case MPCT_IOINT:
			{
				struct mpct_int *m = (struct mpct_int *) mpt;
				mpt += sizeof(*m);
				count += sizeof(*m);
				break;
			}
		case MPCT_LINT:
			{
				struct mpct_int *m = (struct mpct_int *) mpt;
				mpt += sizeof(*m);
				count += sizeof(*m);
				break;
			}
		}
	}
}

const char *APICSystem::getCPUModel(unsigned int cpuID) {
	int family = cpuFamily[cpuID];
	int model = cpuModel[cpuID];
	static char *model_defs[] = {
		"80486DX", "80486DX",
		"80486SX", "80486DX/2 or 80487",
		"80486SL", "Intel5X2(tm)",
		"Unknown", "Unknown",
		"80486DX/4"
	};
	if (family == 0x04 && model < 9) // 100
		return model_defs[model];
	else if (family == 0x5) { // 101
		if (model == 4)
			return ("Pentium with MMX technology");
		else
			return ("Pentium");
	} else if (family == 0x6) { // 110
		if (model == 1)
			return ("Pentium Pro");
		else if (model == 3)
			return ("Pentium II (model 3)");
		else if (model == 5)
			return ("Pentium II (model 5) or Celeron");
		else if (model == 6)
			return ("Celeron");
		else if (model == 7)
			return ("Pentium III (model 7)");
		else if (model == 8)
			return ("Pentium III (model 8) or Celeron");
		else if (model == 10)
			return ("Pentium III Xeon (model A)");
		else
			return ("P6 family");
	} else if (family == 0x0F && model == 0x0F)	// 111
		return ("Special controller");
	else
		return ("Unknown CPU");
}

void APICSystem::delay(unsigned int wraparounds) {
	for (unsigned int i = 0; i < wraparounds; i++) {
		unsigned int curr_count, prev_count = ~0;
		int delta;

		IO_Port port1(0x43);
		IO_Port port2(0x40);
		port1.outb(0x00);
		curr_count = port2.inb();
		curr_count |= port2.inb() << 8;

		do {
			prev_count = curr_count;
			port1.outb(0x00);
			curr_count = port2.inb();
			curr_count |= port2.inb() << 8;
			delta = curr_count - prev_count;

			// Comment from the Linux source code:

			// This limit for delta seems arbitrary, but it isn't, it's
			// slightly above the level of error a buggy Mercury/Neptune
			// chipset timer can cause.

		} while (delta < 300);
	}
}

// the "parameter" for the setup code for the application processors
volatile void *ap_stack;

extern "C" void setup_ap(void);

void APICSystem::initLAPIC() {
    unsigned char llapic_id = (1 << getAndIncNextLogicalAPICID());
    //DBG << "Setting logical APIC ID to " << (int) llapic_id << endl;
    lapic.init();
    lapic.setLogicalLAPICID(llapic_id);

}

/* aktiviert die CPU mit der ID id, gibt bei erfolg true, ansonsten false zurück */
bool APICSystem::bootCPU(unsigned int cpu_id, void *top_of_stack) {
	if (cpu_id >= numberOfCPUs) {
		return false;
	}

	unsigned int id = lapicID[cpu_id];

//	if (id == bspID) {
//		// don't start the BSP, just set the LAPIC's logical destination address
//		initLAPIC();
//		onlineCPUs++;
////		DBG << "CPU " << cpu_id << " is BSP" << endl;
//		return true;
//	}

	ap_stack = top_of_stack; // give top of stack as a "parameter" to setup_ap()
	
	bool IPIDelivered = false;
	unsigned long start_eip = (unsigned long)relocatedSetupAP; // address of relocated setup_ap()

	//DBG << "start_eip = " << hex << start_eip << dec << endl;

	//this should be a *static* assertion
	unsigned char vector = (start_eip >> 12) & 0xff;
	if ((start_eip & ~0x000ff000) ||
	    (vector >= 0xa0 && vector <= 0xbf)) {
//		DBG << "weird start_eip, giving up" << endl;
		return false;
	}

	// set BIOS shutdown code to warm start
	IO_Port port1(0x70);
	IO_Port port2(0x71);
	port1.outb(0xf);
	port2.outb(0xa);
	
	// set reset vector
	*((volatile unsigned short *) 0x469) = start_eip >> 4;
	*((volatile unsigned short *) 0x467) = start_eip & 0xf;

	// reset callout
	calloutVector[id] = 0;

	// assert INIT
	struct ICR_L init_ipi;
	init_ipi.destination_shorthand = DESTINATION_SHORTHAND_NO;
	init_ipi.trigger_mode = TRIGGER_MODE_LEVEL;
	init_ipi.level = LEVEL_ASSERT;
	init_ipi.destination_mode = DESTINATION_MODE_PHYSICAL;
	init_ipi.delivery_mode = DELIVERY_MODE_INIT;
	init_ipi.vector = 0;
	init_ipi.delivery_status = 0; //workaround for qemu
	lapic.sendIPI(id, init_ipi);

	delay(2);
	
	// deassert INIT
	init_ipi.level = LEVEL_DEASSERT;
	lapic.sendIPI(id, init_ipi);

	// if local APIC is external, we don't need STARTUP IPIs
	if (! lapic.isExternalAPIC()) {
		// otherwise we send up to 3 IPIs
		for (int j = 0; j < 2; j++) {
			// send STARTUP IPI
			//DBG << "Sending STARTUP IPI #" << j << endl;
			struct ICR_L startup_ipi;
			startup_ipi.destination_shorthand = DESTINATION_SHORTHAND_NO;
			startup_ipi.trigger_mode = TRIGGER_MODE_EDGE;
			startup_ipi.level = LEVEL_DEASSERT;
			startup_ipi.destination_mode = DESTINATION_MODE_PHYSICAL;
			startup_ipi.delivery_mode = DELIVERY_MODE_STARTUP;
			startup_ipi.vector = (start_eip >> 12);
			startup_ipi.delivery_status = 0; //workaround for qemu
			lapic.sendIPI(id, startup_ipi);
			
			int timeout = 0;
			// wait if STARTUP IPI was successful
			do {
				delay(1);
				timeout++;
				IPIDelivered = lapic.isIPIDelivered();
			} while (!IPIDelivered && (timeout < 10));

			if (IPIDelivered) {
				//DBG << "Successful sending of STARTUP IPI #" << j << endl;
				break; // everything went fine
			}
		}
	}
	if (IPIDelivered) {
		// reset callin
		callinVector[id] = 0;
		//DBG << "Calling out CPU " << id << endl;
		// allow AP to start initialization
		calloutVector[id] = 1;

		//DBG << "Waiting for callin from CPU " << id << endl;
		for (int timeout = 0; timeout < 500; timeout++) {
			if (callinVector[id] == 1) {
				onlineCPUs++;
                //DBG << "Got callin from CPU " << id << endl;
				return true; // CPU has booted
			}
			delay(1);
		}
	}

	//DBG << "Did not get callin from CPU " << id << endl;
	return false; // CPU didn't react
}

void APICSystem::disablePICs()
{
//	DBG << "PIC comp mode, disabling PICs." << endl;
	// write to IMCR
	IO_Port port1(0x22);
	port1.outb(0x70);
	// disable PIC Mode
	IO_Port port2(0x23);
	port2.outb(0x01);
}

/*
 * setup_ap() ist durch einfaches Kopieren relozierbar, da es (ausser beim
 * Sprung in startup_ap()) keine absoluten Adressen verwendet.
 * Die Funktion muss an die Adresse 0x40000 kopiert werden, damit APs dort
 * loslaufen koennen.
 */
extern char ___SETUP_AP_START__, ___SETUP_AP_END__;
void APICSystem::copySetupAPtoLowMem() {
    char *src = &___SETUP_AP_START__, *dest = (char *) 0x40000;
    relocatedSetupAP = dest;
    while (src < &___SETUP_AP_END__) {
        *dest++ = *src++;
    }
}

void APICSystem::sendCustomIPI(unsigned char logicalDestination, unsigned char vector) {
	struct ICR_L data; 
	data.vector = vector;
	data.delivery_mode = DELIVERY_MODE_FIXED;
	data.destination_mode = DESTINATION_MODE_LOGICAL;
	data.level = LEVEL_ASSERT;
	data.destination_shorthand = DESTINATION_SHORTHAND_NO;

	lapic.sendIPI(logicalDestination, data); 
}

