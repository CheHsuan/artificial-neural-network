cc ?= gcc
CFLAGS = -O0 -Wall -std=c99
LDFLAGS= -lm 

EXEC = model

all: $(EXEC)

$(EXEC): ./src/matrix.c ./src/neuralnet.c ./src/xmlparser.c ./src/dataprocessing.c main.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	$(RM) $(EXEC)
