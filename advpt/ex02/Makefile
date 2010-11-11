#==================================================================================================
#
#  Advanced Programming Techniques (AdvPT)
#  Winter Term 2010
#  Assignment 2 - Makefile
#
#==================================================================================================


CXX = g++
CXXFLAGS = -Wall -Werror -Wshadow -ansi -pedantic

SRC = $(wildcard ./*.cpp)
BIN = $(SRC:.cpp=)

default: $(BIN)

clean:
	$(RM) $(BIN)
