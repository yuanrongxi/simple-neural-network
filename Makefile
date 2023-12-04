c_file		=$(wildcard ./src/*.c)

o_file		=$(patsubst %.c,%.o,$(c_file))
d_file		=$(patsubst %.c,%.d,$(c_file))


CURRENT_DIR 	:=$(shell pwd)
CC		=gcc

INCLUDE_DIR :=	$(CURRENT_DIR) \
				/usr/local/include

CC_FLAGS	=  -Wall -O -g -fopenmp $(addprefix -I, $(INCLUDE_DIR))  -D_POSIX_C_SOURCE=200809L
LD_FLAGS	=  -lm
CC_DEPFLAGS	=-MMD -MF $(@:.o=.d) -MT $@
TARGET_NN	= nn

all: print_c print_o nn

print_c:
	echo $(c_file)
print_o:
	echo $(o_file)

%.o:  %.c
	$(CC)  $(CC_FLAGS) $(CC_DEPFLAGS) -c $< -o $@

nn: $(o_file)
	$(CC) $(CC_FLAGS) -o $(TARGET_NN) $(o_file) $(LD_FLAGS)

.PHONY :clean

clean:
	rm -f $(o_file) $(d_file) nn


-include $(wildcard $(o_file:.o=.d))
