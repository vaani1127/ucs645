CC = mpicc
CFLAGS = -O2

all: serial parallel

serial:
	gcc -O2 -o q1_serial q1_serial.c

parallel:
	$(CC) $(CFLAGS) -o q1_parallel q1_parallel.c

clean:
	rm -f q1_serial q1_parallel