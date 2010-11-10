# $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
# kate: encoding ISO-8859-15;
# vim: set fileencoding=latin-9:
# -*- coding: latin-9 -*-


all: bootsect setup

% : %.asm
	nasm -f bin $<

clean:
	rm -f *~ bootsect setup
