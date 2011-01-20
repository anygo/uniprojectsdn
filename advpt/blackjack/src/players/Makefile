#==================================================================================================
#
#  Advanced Programming Techniques (AdvPT)
#  Winter Term 2009/2010
#  Project phase - Black Jack
#
#  Copyright (C) 2010 Klaus Iglberger
#
#==================================================================================================

# Includes
CXXFLAGS += -I$(INSTALL_PATH)

# Setting the source, object and dependency files
SRC = $(wildcard ./*.cpp)
OBJ = $(SRC:.cpp=.o)
DEP = $(SRC:.cpp=.d)

# Rules
default: $(OBJ)
	@$(foreach dir,$(OBJ),cp -p $(dir) $(OBJECT_PATH)/$(notdir $(dir));)

clean:
	@$(RM) $(OBJ) $(DEP)

# Makefile includes
ifneq ($(MAKECMDGOALS),clean)
-include $(DEP)
endif

# Makefile generation
%.d: %.cpp
	@set -e; rm -f $@; \
	$(CXX) -MM -MT $*.o $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	$(RM) $@.$$$$

# Setting the independent commands
.PHONY: default clean
