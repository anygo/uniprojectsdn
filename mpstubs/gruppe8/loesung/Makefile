# $Date: 2009-08-25 10:51:40 +0200 (Tue, 25 Aug 2009) $, $Revision: 2214 $, $Author: benjamin $
# kate: encoding ISO-8859-15;
# vim: set fileencoding=latin-9:
# -*- coding: latin-9 -*-

# -----------------------------------------------------------------------------
# 
#                              M A K E F I L E
# 
# zum Uebungsbetriebssystem OOStuBS der Lehrveranstaltung 'Betriebssysteme'.
# -----------------------------------------------------------------------------
#
# Durch Eingabe von 'make' werden die Quelltexte des Systems kompiliert. 'make
# bootdisk' schreibt das System-Image auf eine Diskette, von der das System
# dann gebootet werden kann. Mit 'make clean' wird im Verzeichnis aufgeraeumt.
# Die Targets 'bochs', 'bochs-gdb', 'gdb', 'ddd' starten den Bochs-Emulator mit
# bzw. ohne GDB-Stub und starten den GDB- bzw. den DDD-Debugger, um sich dann
# mit dem GDB-Stub zu verbinden.
#
# -----------------------------------------------------------------------------
# Rewrite 05/2008: wanja
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Liste der Quelltexte:

STARTUP_SOURCE = ./startup.asm ./setup_ap.asm
CC_SOURCES = $(shell find . -name "*.cc")
C_SOURCES = $(shell find . -name "*.c")
ASM_SOURCES = $(shell find ./machine -name "*.asm")

# Einstellungen in Abhaengigkeit vom Generierungssystem:
#
#    ASMOBJFORMAT: Ausgabeformat fuer den Assembler. Das Format muss dem
#                  jeweiligen Format des verwendeten C++ Compilers angepasst
#                  werde, damit gemischte Objektdateien gelinkt werden koennen.
#    OBJDIR:       Verzeichnis, in dem die Objektdateien abgelegt werden
#                  sollen.
#    DEPDIR:       Verzeichnis, in dem die Abhaengigkeitsdateien abgelegt werden
#                  sollen
#    DRIVE:        Laufwerk, auf das das System-Image geschrieben wird
#    DELETE:       Name des Kommandos zum Loeschen von Dateien
#    ASM:          Zu benutzender Assembler
#    CC/CXX:       Zu benutzender C/C++-Compiler
#    CFLAGS:       Flags fuer den C-Compileraufruf
#    CXXFLAGS:     Flags fuer den C++-Compileraufruf
#    LD:           Zu benutzender Linker
#    LDFLAGS:      Flags fuer den Linkeraufruf
#    LDLIBS:       Dazuzulinkende Bibliotheken
#    LDHEAD:       Am Anfang zu linkende Dateien
#    LDTAIL:       Am Ende zu linkende Dateien
#    BOOT:         Das Verzeichnis zu Bootsektor und Setup-Code
#    TOOLS:        Das Verzeichnis mit dem Build-Tool

VERBOSE = @
ASMOBJFORMAT = elf
OBJDIR = ./build
DEPDIR = ./dep
DRIVE = /dev/fd0
DELETE = rm
ASM = nasm
CC ?= gcc
CXX ?= g++
CFLAGS := $(CFLAGS) -m32 -Wall -I. -g -O3
CXXFLAGS := $(CFLAGS) -Wno-non-virtual-dtor -fno-rtti -fno-exceptions -Wno-write-strings
LD ?= ld
LDFLAGS := -melf_i386
ifeq ($(shell uname),SunOS)
LDHEAD = /proj/i4bs/i386/lib/crti.o $(shell g++ --print-file-name=crtbegin.o)
LDTAIL = $(shell g++ --print-file-name=crtend.o) /proj/i4bs/i386/lib/crtn.o
else
LDHEAD = $(shell g++ -m32 --print-file-name=crti.o && g++ -m32 --print-file-name=crtbegin.o)
LDTAIL = $(shell g++ -m32 --print-file-name=crtend.o && g++ -m32 --print-file-name=crtn.o)
endif
BOOT = ../boot
TOOLS = ../tools
BOOTDEVICE = 0:80:2:18:0:0:1

# -------------------------------------------------------------------------
# Namen der Unterverzeichnisse mit den Quelltexten

VPATH = $(sort $(dir $(STARTUP_SOURCE) $(CC_SOURCES) $(C_SOURCES) $(ASM_SOURCES)))

# -------------------------------------------------------------------------
# Listen mit den Objektdateien, die beim Kompilieren entstehen:

FIRST_OBJECT = $(addprefix $(OBJDIR)/,$(patsubst %.asm,_%.o, $(notdir $(STARTUP_SOURCE))))
C_OBJECTS = $(notdir $(C_SOURCES:.c=.o))
CC_OBJECTS = $(notdir $(CC_SOURCES:.cc=.o))

DEP_FILES = $(patsubst %.o,$(DEPDIR)/%.d,$(C_OBJECTS))
DEP_FILES += $(patsubst %.o,$(DEPDIR)/%.d,$(CC_OBJECTS))

ASM_OBJECTS = $(patsubst %.asm,_%.o, $(notdir $(ASM_SOURCES)))
OBJPRE = $(addprefix $(OBJDIR)/,$(ASM_OBJECTS) $(C_OBJECTS) $(CC_OBJECTS))

# --------------------------------------------------------------------------
# Default target

all: $(OBJDIR)/bootdisk.qemu

# --------------------------------------------------------------------------
# Regeln zur Erzeugung der Abhaengigkeitsdateien

$(DEPDIR)/%.d : %.c
	@echo "DEP		$@"
	@if test \( ! \( -d $(@D) \) \) ;then mkdir -p $(@D);fi
	$(VERBOSE) $(CC) $(CFLAGS) -MM -MT $(OBJDIR)/$*.o -MF $@ $<

$(DEPDIR)/%.d : %.cc
	@echo "DEP		$@"
	@if test \( ! \( -d $(@D) \) \) ;then mkdir -p $(@D);fi
	$(VERBOSE) $(CXX) $(CFLAGS) -MM -MT $(OBJDIR)/$*.o -MF $@ $<

# --------------------------------------------------------------------------
# Regeln zur Erzeugung der Objektdateien

$(OBJDIR)/%.o : %.c
	@echo "CC		$@"
	@if test \( ! \( -d $(@D) \) \) ;then mkdir -p $(@D);fi
	$(VERBOSE) $(CC) -c $(CFLAGS) -o $@ $<

$(OBJDIR)/%.o : %.cc
	@echo "CXX		$@"
	@if test \( ! \( -d $(@D) \) \) ;then mkdir -p $(@D);fi
	$(VERBOSE) $(CXX) -c $(CXXFLAGS) -o $@ $<

$(OBJDIR)/_%.o : %.asm
	@echo "ASM		$@"
	@if test \( ! \( -d $(@D) \) \) ;then mkdir -p $(@D);fi
	$(VERBOSE) $(ASM) -f $(ASMOBJFORMAT) -o $@ $<

# --------------------------------------------------------------------------
# Gelinktes System

$(OBJDIR)/system: $(FIRST_OBJECT) $(OBJPRE)
	@echo "LD		$@"
	@if test \( ! \( -d $(@D) \) \) ;then mkdir -p $(@D);fi
	$(VERBOSE) $(LD) -e startup -T sections -o $(OBJDIR)/system $(LDFLAGS) $(FIRST_OBJECT) $(LDHEAD) $(OBJPRE) $(LDTAIL) $(LDLIBS)

# --------------------------------------------------------------------------
# 'system.img' enthaelt die zu ladenden Sections des eigentlichen Systems

$(OBJDIR)/system.img : $(OBJDIR)/system
	@echo "OBJCOPY		$@"
	@if test \( ! \( -d $(@D) \) \) ;then mkdir -p $(@D);fi
	$(VERBOSE) objcopy -O binary $< $@

# --------------------------------------------------------------------------
# 'bootdisk.img' besteht aus dem 'system.img', das um den (parametrisierten)
# Bootblock und den Setup-Code erweitert wurde.

$(OBJDIR)/bootdisk.img : $(OBJDIR)/system.img $(TOOLS)/build $(BOOT)/bootsect $(BOOT)/setup
	@echo "BUILD		$@"
	@if test \( ! \( -d $(@D) \) \) ;then mkdir -p $(@D);fi
	$(VERBOSE) $(TOOLS)/build $(BOOT)/bootsect $(BOOT)/setup $< $(BOOTDEVICE) $@

# --------------------------------------------------------------------------
# 'bootdisk.qemu' ist ein mit Nullen auf 1,44 MB aufgefuelltes Image,
# so wie es qemu benoetigt

$(OBJDIR)/bootdisk.qemu : $(OBJDIR)/bootdisk.img
	@echo "DD		$@"
	@if test \( ! \( -d $(@D) \) \) ;then mkdir -p $(@D);fi
	$(VERBOSE) dd if=$< of=$@ bs=1474560 conv=sync

# --------------------------------------------------------------------------
# 'clean' loescht das generierte System, die Objektdateien und die
# Abhaengigkeitsdateien

clean:	
	@echo "RM		$(OBJDIR)"
	$(VERBOSE) rm -rf $(OBJDIR)
	@echo "RM		$(DEPDIR)"
	$(VERBOSE) rm -rf $(DEPDIR)
	$(MAKE) -C $(BOOT) clean

# --------------------------------------------------------------------------
# 'bootdisk' erzeugt zunaechst das System, falls das noch nicht geschehen ist.
# Danach wird eine Bootdiskette mit dem System erzeugt.

bootdisk: $(OBJDIR)/bootdisk.img
	@echo "CP		$<"
	$(VERBOSE) cp $(OBJDIR)/bootdisk.img $(DRIVE)

# --------------------------------------------------------------------------
# 'qemu' ruft den qemu-Emulator mit dem System auf. Es wird eine zwei-CPU-
# Maschine emuliert. (Stört bei OOStuBS nicht weiter)

qemu: $(OBJDIR)/bootdisk.qemu
	/proj/i4bs/qemu/bin/qemu -fda $(OBJDIR)/bootdisk.qemu -boot a -k en-us -smp 2

# --------------------------------------------------------------------------
# 'qemu-gdb' ruft den qemu-Emulator mit aktiviertem GDB-Stub mit dem System
# auf, sodass es per GDB oder DDD inspiziert werden kann.

qemu-gdb: $(OBJDIR)/bootdisk.qemu
	rm -f /tmp/gdbcommands.$(shell id -u)
	$(VERBOSE) echo "target remote localhost:$(shell id -u)" >> /tmp/gdbcommands.$(shell id -u)
	/peoj/i4bs/qemu/bin/qemu -p $(shell id -u) -fda $(OBJDIR)/bootdisk.qemu -boot a -k en-us -smp 2 -s -S &

# --------------------------------------------------------------------------
# 'gdb' startet den GDB-Debugger und verbindet sich mit dem GDB-Stub des vorher
# gestarteten Bochs/qemu.

gdb:
	gdb -x /tmp/gdbcommands.$(shell id -u) $(OBJDIR)/system

# 'ddd' startet den DDD-Debugger und verbindet sich mit dem GDB-Stub des vorher
# gestarteten Bochs/qemu.

ddd:
	ddd --gdb -x /tmp/gdbcommands.$(shell id -u) $(OBJDIR)/system

# --------------------------------------------------------------------------
# Das Image wirds ins Verzeichnis des tftp-servers kopiert, um es ueber
# pxelinux auf den Testrechnern starten zu können.

netboot: $(OBJDIR)/system
	cp $(OBJDIR)/system /proj/i4bs/tftpboot/kernels/$(shell id -run)/kernel	

# --------------------------------------------------------------------------
# 'build'-Tool kompilieren.

$(TOOLS)/build: $(TOOLS)/build.c
	cd $(TOOLS) && $(CC) -o $@ $<

# --------------------------------------------------------------------------
# Bootsektor und Protected-Mode-Setup-Code kompilieren.

$(BOOT)/bootsect $(BOOT)/setup: $(BOOT)/bootsect.asm $(BOOT)/setup.asm
	$(MAKE) -C $(BOOT)

# --------------------------------------------------------------------------
# Einbindung der Abhaengigkeitsdateien

ifneq ($(MAKECMDGOALS),clean)
-include $(DEP_FILES)
endif

.PHONY: clean bootdisk qemu qemu-gdb gdb ddd
