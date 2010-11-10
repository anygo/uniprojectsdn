// $Date: 2009-08-11 16:57:46 +0200 (Tue, 11 Aug 2009) $, $Revision: 2208 $, $Author: benjamin $
// kate: encoding ISO-8859-15;
// vim: set fileencoding=latin-9:
// -*- mode: c; coding: latin-9 -*- 

#include "object/o_stream.h"

/* Hier muesst ihr selbst Code vervollstaendigen */ 

O_Stream::O_Stream(){
	base = 10;
}

O_Stream::~O_Stream(){
}

O_Stream &O_Stream::operator<<(char c) {
	put(c);
	return *this;
}

O_Stream &O_Stream::operator<<(unsigned char c) {
	put(c);
	return *this;
}

O_Stream &O_Stream::operator<<(char* string) {

	while (*string != '\0') {
		put(*string);
		string++;
	}
	return *this;
}

O_Stream &O_Stream::operator<<(short ival) {
	return *this << (long) ival;

}

O_Stream &O_Stream::operator<<(unsigned short ival) {
	return *this << (unsigned long) ival;

}

O_Stream &O_Stream::operator<<(int ival) {
	return *this << (long) ival;

}

O_Stream &O_Stream::operator<<(unsigned int ival) {
	return *this << (unsigned long) ival;
}

O_Stream &O_Stream::operator<<(long ival) {
	if (ival < 0) {
		if (base == 10) {
			(*this) << '-';
			return (*this) << (unsigned long) (-ival);
		}
	}
	return (*this) << (unsigned long) ival;
}

O_Stream &O_Stream::operator<<(unsigned long ival) {
	int i = 0;
	short num;
	char c = ' '; 
	
	if (ival == 0) {
		tmp[i++] = '0';
	}
	while (ival > 0) {
		num = ival % base;
		
		switch (num) {
			case 0: c = '0'; break;
			case 1: c = '1'; break;
			case 2: c = '2'; break;
			case 3: c = '3'; break;
			case 4: c = '4'; break;
			case 5: c = '5'; break;
			case 6: c = '6'; break;
			case 7: c = '7'; break;
			case 8: c = '8'; break;
			case 9: c = '9'; break;
			case 10: c = 'A'; break;
			case 11: c = 'B'; break;
			case 12: c = 'C'; break; 
			case 13: c = 'D'; break;
			case 14: c = 'E'; break;
			case 15: c = 'F'; break;
		}

		tmp[i++] = c;
		ival /= base;
	}
	for (i=i-1; i >= 0; i--) {
		put(tmp[i]);
	}
	return *this;

}

O_Stream &O_Stream::operator<<(void *ptr) {
	long ival = (long)ptr;
	
	return (*this) << ival;

}

O_Stream &O_Stream::operator<<(O_Stream &(*f)(O_Stream &) ) {
	return f(*this);
}

O_Stream &hex(O_Stream &os) {
	os.base = 16;
	return os;
}
O_Stream &dec(O_Stream &os) {
	os.base = 10;
	return os;
}
O_Stream &oct(O_Stream &os) {
	os.base = 8;
	return os;
}
O_Stream &bin(O_Stream &os) {
	os.base = 2;
	return os;
}
O_Stream &endl(O_Stream &os) {
	os << '\n';
	os.flush();
	return os;
}
