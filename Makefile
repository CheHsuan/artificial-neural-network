EXEC = annmodel
all: $(EXEC)

cc ?= gcc
CFLAGS = -O0 -Wall -std=c99 -D__forceinline="__attribute__((always_inline))" -fopenmp
LDFLAGS = -lm -lpthread 
PROF_FLAGS = -pg

CFLAGS += $(PROF_FLAGS)

$(EXEC): ./src/neuralnet.c ./src/xmlparser.c ./src/threadpool.c ./src/dataprocessing.c main.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	$(RM) $(EXEC) gmon.out
