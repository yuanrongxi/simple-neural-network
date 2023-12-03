#ifndef __nl_array_h
#define __nl_array_h

#include "nl_common.h"

typedef struct {
	float* data;
	int size;

	int row;
	int col;
}nl_array_t;

nl_array_t* nl_create_array(int col, int row);
void nl_set_array(nl_array_t*a, float*data, int col, int row);
void nl_free_array(nl_array_t* a);

void nl_array_zero_reshape(nl_array_t* a, int col, int row);
void nl_array_zero(nl_array_t* a);
void nl_array_reshape(nl_array_t* a, int col, int row);

nl_array_t* nl_array_transpose(nl_array_t* a, nl_array_t* b);

void nl_array_set_val(nl_array_t* a, int r, int c, float v);

nl_array_t* nl_array_randn(int col, int row);
	
nl_array_t* nl_array_add_val(nl_array_t* arr, float val);
nl_array_t* nl_array_sub_val(nl_array_t* arr, float val);
nl_array_t* nl_array_mul_val(nl_array_t* arr, float val);
nl_array_t* nl_array_div_val(nl_array_t* arr, float val);

nl_array_t* nl_array_add(nl_array_t* dst, nl_array_t* first, nl_array_t* second);
nl_array_t* nl_array_add_self(nl_array_t* dst, nl_array_t* src);
nl_array_t* nl_array_sub(nl_array_t* dst, nl_array_t* first, nl_array_t* second);
nl_array_t* nl_array_sub_self(nl_array_t* dst, nl_array_t* src);

nl_array_t* nl_array_mul(nl_array_t* dst, nl_array_t* first, nl_array_t* second);
nl_array_t* nl_array_mul_self(nl_array_t* dst, nl_array_t* src);
nl_array_t* nl_array_div(nl_array_t* dst, nl_array_t* first, nl_array_t* second);
nl_array_t* nl_array_div_self(nl_array_t* dst, nl_array_t* src);

nl_array_t* nl_array_merge_delta(nl_array_t* dst, nl_array_t* delta, float f);

nl_array_t* nl_array_dot(nl_array_t* dst, nl_array_t* first, nl_array_t* second);
nl_array_t* nl_array_first_T_dot(nl_array_t* dst, nl_array_t* first, nl_array_t* second);
nl_array_t* nl_array_second_T_dot(nl_array_t* dst, nl_array_t* first, nl_array_t* second);


nl_array_t* nl_array_sigmoid(nl_array_t* a);
nl_array_t* nl_array_sigmoid_prime(nl_array_t* a);
/*input sigmoid array*/
nl_array_t* nl_array_prime(nl_array_t* a);

nl_array_t* nl_array_relu(nl_array_t* s);
nl_array_t* nl_array_relu_grad(nl_array_t* in, nl_array_t* delta);

nl_array_t* nl_array_softmax(nl_array_t* d, nl_array_t* s);

nl_array_t* nl_array_conv(nl_array_t* d, nl_array_t* s, nl_array_t* filter, float b);
nl_array_t* nl_array_conv_grad(nl_array_t* d, nl_array_t* delta_output, nl_array_t* image);

nl_array_t* nl_array_pooling(nl_array_t* d, nl_array_t* s, int pooling);
nl_array_t* nl_array_pooling_grad(nl_array_t* d, nl_array_t* delta_output, int pooling);

int nl_array_argmax(nl_array_t* a);

void nl_array_log(nl_array_t* a);

#endif



