CC=gcc
CFLAGS=-Wall -g -I.

OBJS=bench.o nr_pdcp_entity.o nr_pdcp_integrity_nia2.o nr_pdcp_security_nea2.o \
     nr_pdcp_sdu.o aes_128_cbc_cmac.o aes_128_ctr.o

OBJS_optimized=bench.o nr_pdcp_entity.o nr_pdcp_integrity_nia2_optimized.o \
     nr_pdcp_security_nea2_optimized.o nr_pdcp_sdu.o aes_128_cbc_cmac.o \
     aes_128_ctr.o

bench: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lcrypto

all: bench bench_optimized

bench_optimized: $(OBJS_optimized)
	$(CC) $(CFLAGS) -o $@ $(OBJS_optimized) -lcrypto

aes_128_ctr.o: ../../../../openair3/SECU/aes_128_ctr.c
	$(CC) $(CFLAGS) -c -o $@ $<

aes_128_cbc_cmac.o: ../../../../openair3/SECU/aes_128_cbc_cmac.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: ../%.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f bench $(OBJS) bench_optimized $(OBJS_optimized)
