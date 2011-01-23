#==================================================================================================
#
#  Advanced Programming Techniques (AdvPT)
#  Winter Term 2009/2010
#  Project phase - Black Jack
#
#  Copyright (C) 2010 Klaus Iglberger
#
#==================================================================================================

# Compiler configuration
export CXX      = g++
export CXXFLAGS = -Wall -Werror -Wextra -Winline -Wshadow -Woverloaded-virtual -ansi -pedantic --param inline-unit-growth=150 --param max-inline-insns-single=500 --param large-function-growth=600 -g -I$(CURDIR)

# Exporting the installation path
export INSTALL_PATH = $(CURDIR)
export OBJECT_PATH  = $(CURDIR)/obj

# Rules
default: core players
	@echo "Compiling the executable..."
	$(CXX) $(CXXFLAGS) -o play Main.cpp $(OBJECT_PATH)/*.o
	@echo "...Compilation of the executable complete!"
	@echo

core:
	@echo
	@echo "Setup of the core module..."
	@$(MAKE) --no-print-directory -C ./src/core/
	@echo "...Setup of the core module complete!"
	@echo

players:
	@echo "Setup of the players..."
	@$(MAKE) --no-print-directory -C ./src/players
	@echo "...Setup of the players complete!"
	@echo


# Clean up
clean:
	@$(MAKE) --no-print-directory -C ./src/core/ clean
	@$(MAKE) --no-print-directory -C ./src/players/ clean
	@$(RM) play ./obj/*.o
