// $Date: 2009-08-11 16:57:46 +0200 (Tue, 11 Aug 2009) $, $Revision: 2208 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#include "machine/acpi.h"

namespace ACPI_Subsystem {

bool sums_to_zero( void *start, int len) {
       	int sum = 0;
	char* start_addr = (char *) start;
	if ( len == 0)
		return false;
       	while (len--)
		sum += *start_addr++;
	return !(sum & 0xff);
}

//spec 5.2.5.1
//maybe doesn't work for all systems (only EISA/MCA systems according to spec)
RSDP *RSDP::find()
{
	void *ebda_ptr = *( (void **) ( (0x40<<4)+0x0e) );
	void *bios_ro_memspace = (void *) (0xe0000);
	RSDP *found;

	if ( ( found = find( ebda_ptr, 0x400 ) ) )
		return found;

	if ( ( found = find( bios_ro_memspace, 0x20000 ) ) )
		return found;

//	DBG << "ACPI: No root system description pointer found" << endl;
	return 0;
}

RSDP *RSDP::find( void *start, int len )
{
	int *search = (int *) start;
	int *end = ( search + len / 4);
	RSDP *found;

	for (; search < end; search+=4 )
	{
		
		if ( search[0] != CHARS_TO_UINT32('R','S','D',' ') )
			continue;

		if ( search[1] != CHARS_TO_UINT32('P','T','R',' ') ) 
			continue;

		found = (RSDP *) search;
		if ( found->check_sum() )
			return found;

//		DBG << "ACPI: found RDSP signature at" << (void *) found
//		    << ", but RDSP checksum is wrong" << endl;
	}

	return 0;
}

} // ACPI_Subsystem

