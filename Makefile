EXEC = annmodel
all: $(EXEC)

cc ?= gcc
CFLAGS = -O0 -Wall -std=c99
LDFLAGS = -lm 
PROF_FLAGS = -pg

CFLAGS += $(PROF_FLAGS)

$(EXEC): ./src/matrix.c ./src/neuralnet.c ./src/xmlparser.c ./src/dataprocessing.c main.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	$(RM) $(EXEC) gmon.out
