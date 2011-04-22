CC       = /usr/bin/g++
CPPFLAGS = -I/usr/include/opencv
LDFLAGS  = -L/usr/lib -lcv -lhighgui -lml

OBJ = main.o decaptcha_tool.o captcha_processor.o
BIN = decaptcha

prog: $(OBJ)
	$(CC) $(CPPFLAGS) -o $(BIN) $(OBJ) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<


.PHONY: clean

clean:
	rm -rf $(OBJ) $(BIN)
