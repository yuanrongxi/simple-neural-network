#include <math.h>
#include "nl_array.h"
#include "nl_guass_rand.h"

nl_array_t* nl_create_array(int col, int row) {
	nl_array_t* a = (nl_array_t*)malloc(sizeof(nl_array_t));
	a->col = col;
	a->row = row;
	a->size = col * row;

	a->data = (float*)calloc(a->size, sizeof(float));

	return a;
}

void nl_set_array(nl_array_t*a, float* data, int col, int row) {
	a->col = col;
	a->row = row;
	a->size = col * row;
	a->data = data;
}

nl_array_t* nl_array_transpose(nl_array_t* a, nl_array_t* b) {
	a->size = b->size;
	a->col = b->row;
	a->row = b->col;

	float* ps = a->data;
	for (int i = 0; i < a->row; i++) {
		for (int j = 0; j < a->col; j++){
			*ps ++ = b->data[j * b->col + i];
		}
	}

	return a;
}

void nl_free_array(nl_array_t* a) {
	if (a) {
		if (a->data != NULL) 
			free(a->data);
		free(a);
	}
}

static inline bool is_valid_array(nl_array_t* a) {
	if (a->data == NULL ||a->size == 0 || a->col == 0 || a->row == 0) {
		printf("invalid array, size:%d, row:%d, col:%d\n", a->size, a->row, a->col);
		return false;
	}
	return true;
}

static inline bool is_same_shape(nl_array_t* a, nl_array_t* b) {
	return (a->col == b->col && a->row == b->row);
}

nl_array_t* nl_array_randn(int col, int row) {
	nl_array_t* a = nl_create_array(col, row);

	float* data = (float*)a->data;
	int c, r, i = 0;
	for (r = 0; r < a->row; r++) {
		for (c = 0; c < a->col; c++) {
			data[i++] = nl_guass_rand();
		}
	}

	return a;
}

nl_array_t* nl_array_add_val(nl_array_t* a, float val) {

	if (!is_valid_array(a) || val == 0.0f)
		return a;

	float* data = (float*)a->data;
	int i;
	#pragma omp parallel for
	for (i = 0; i < a->size; i++)
		data[i] += val;

	return a;
}

nl_array_t* nl_array_sub_val(nl_array_t* a, float val) {
	if (!is_valid_array(a) || val == 0.0f)
		return a;

	float* data = (float*)a->data;
	int i;
	#pragma omp parallel for
	for (i = 0; i < a->size; i++)
		data[i] -= val;

	return a;
}

nl_array_t* nl_array_mul_val(nl_array_t* a, float val) {
	if (!is_valid_array(a))
		return a;

	float* data = (float*)a->data;
	int i;
	#pragma omp parallel for
	for (i = 0; i < a->size; i++)
		data[i] *= val;

	return a;
}

nl_array_t* nl_array_div_val(nl_array_t* a, float val) {
	if (!is_valid_array(a) || val == 0.0f)
		return a;

	float* data = (float*)a->data;
	int i;
	#pragma omp parallel for
	for (i = 0; i < a->size; i++)
		data[i] /= val;

	return a;
}

nl_array_t* nl_array_add(nl_array_t* dst, nl_array_t* first, nl_array_t* second) {
	if (!is_same_shape(dst, first) || !is_same_shape(dst, second)) {
		printf("array_add shape error, dst (%d,%d), first(%d,%d), second(%d,%d)\n",
				dst->col, dst->row, 
				first->col, first->row,
				second->col, second->row);
		abort();
	}
	
	int i;
	#pragma omp parallel for
	for (i = 0; i < dst->size; i++) {
		dst->data[i] = first->data[i] + second->data[i];
	}

	return dst;
}

nl_array_t* nl_array_add_self(nl_array_t* dst, nl_array_t* src) {
	return nl_array_add(dst, dst, src);
}

nl_array_t* nl_array_sub(nl_array_t* dst, nl_array_t* first, nl_array_t* second) {
	if (!is_same_shape(dst, first) || !is_same_shape(dst, second)) {
		printf("array_sub shape error, dst shape(%d,%d), first shape(%d,%d), second shape(%d,%d)\n",
			dst->col, dst->row,
			first->col, first->row,
			second->col, second->row);
		abort();
	}

	int i;
	#pragma omp parallel for
	for (i = 0; i < dst->size; i++)
		dst->data[i] = first->data[i] - second->data[i];

	return dst;
}

nl_array_t* nl_array_sub_self(nl_array_t* dst, nl_array_t* src) {
	return nl_array_sub(dst, dst, src);
}

nl_array_t* nl_array_mul(nl_array_t* dst, nl_array_t* first, nl_array_t* second) {
	if (!is_same_shape(dst, first) || !is_same_shape(dst, second)) {
		printf("array_mul shape error, dst shape(%d,%d), first shape(%d,%d), second shape(%d,%d)\n",
			dst->col, dst->row,
			first->col, first->row,
			second->col, second->row);
		abort();
	}

	int i;
	#pragma omp parallel for
	for (i = 0; i < dst->size; i++)
		dst->data[i] = first->data[i] * second->data[i];

	return dst;
}

nl_array_t* nl_array_mul_self(nl_array_t* dst, nl_array_t* src) {
	return nl_array_mul(dst, dst, src);
}

nl_array_t* nl_array_div(nl_array_t* dst, nl_array_t* first, nl_array_t* second) {
	if (!is_same_shape(dst, first) || !is_same_shape(dst, second)) {
		printf("array_div shape error, dst shape(%d,%d), first shape(%d,%d), second shape(%d,%d)\n",
			dst->col, dst->row,
			first->col, first->row,
			second->col, second->row);
		abort();
	}

	int i;
	#pragma omp parallel for
	for (i = 0; i < first->size; i++)
		dst->data[i] = first->data[i] / second->data[i];

	return dst;
}

nl_array_t* nl_array_div_self(nl_array_t* dst, nl_array_t* src) {
	return nl_array_div(dst, dst, src);
}

nl_array_t* nl_array_merge_delta(nl_array_t* dst, nl_array_t* delta, float f) {
	int i;
	#pragma omp parallel for
	for (i = 0; i < dst->size; i++) {
		dst->data[i] -= f * delta->data[i];
		delta->data[i] = 0;
	}

	return dst;
}

nl_array_t* nl_array_dot(nl_array_t* dst, nl_array_t* first, nl_array_t* second) {
	if (first->col != second->row) {
		printf("array can't dot, dst size:%d, first col:%d, second row:%d\n", 
					dst->size, first->col, second->row);
		abort();
	}

	dst->row = first->row;
	dst->col = second->col;
	dst->size = dst->row * dst->col;

	int r;
	#pragma omp parallel for private(r)
	for (r = 0; r < first->row; r++) {
		for (int c = 0; c < second->col; c++){
			float s = 0.0f;
			for (int i = 0; i < first->col; i++) {
				s += first->data[i + first->col * r] * second->data[i * second->col + c];
			}
			dst->data[r * second->col + c] = s;
		}
	}

	return dst;
}

nl_array_t* nl_array_first_T_dot(nl_array_t* dst, nl_array_t* first, nl_array_t* second) {
	if (first->row != second->row) {
		printf("first T array can't dot, dst size:%d, first row:%d, second row:%d\n",
			dst->size, first->row, second->row);
		abort();
	}

	dst->row = first->col;
	dst->col = second->col;
	dst->size = dst->row * dst->col;

	int r;
	#pragma omp parallel for private(r)
	for (r = 0; r < first->col; r++) {
		for (int c = 0; c < second->col; c++){
			float s = 0.0f;
			for (int i = 0; i < first->row; i++) {
				s += first->data[i*first->col + r] * second->data[i * second->col + c];
			}
			dst->data[r * second->col + c] = s;
		}
	}

	return dst;
}

nl_array_t* nl_array_second_T_dot(nl_array_t* dst, nl_array_t* first, nl_array_t* second) {
	if (first->col != second->col) {
		printf("array can't dot, dst size:%d, first col:%d, second row:%d\n",
			dst->size, first->col, second->row);
		abort();
	}

	dst->row = first->row;
	dst->col = second->row;
	dst->size = dst->row * dst->col;

	int r;
	#pragma omp parallel for private(r)
	for (r = 0; r < first->row; r++) {
		for (int c = 0; c < second->row; c++){
			float s = 0.0f;
			for (int i = 0; i < first->col; i++) {
				s += first->data[i + first->col * r] * second->data[i + c * second->col];
			}
			dst->data[r * second->row + c] = s;
		}
	}

	return dst;
}

static inline float sigmoid(float x) {
	return (1.0f /(1.0f + expf(-x)));
}

nl_array_t* nl_array_sigmoid(nl_array_t* a) {
	int i;
	#pragma omp parallel for
	for (i = 0; i < a->size; i++) {
		a->data[i] = sigmoid(a->data[i]);
	}

	return a;
}

nl_array_t* nl_array_sigmoid_prime(nl_array_t* a) {
	int i;
	#pragma omp parallel for	
	for (i = 0; i < a->size; i++) {
		float z = sigmoid(a->data[i]);
		a->data[i] = z * (1 - z);
	}

	return a;
}


nl_array_t* nl_array_prime(nl_array_t* a) {
	int i;
	#pragma omp parallel for	
	for (i = 0; i < a->size; i++) {
		a->data[i] = a->data[i] * (1 - a->data[i]);
	}
	return a;
}

int nl_array_argmax(nl_array_t* a) {
	int max_i = 0;
	float max_val = a->data[0];
	for (int i = 1; i < a->size; i++) {
		if (max_val < a->data[i]) {
			max_val = a->data[i];
			max_i = i;
		}
	}

	return max_i;
}

void nl_array_zero_reshape(nl_array_t* a, int col, int row) {
	nl_array_reshape(a, col, row);
	nl_array_zero(a);
}

void nl_array_zero(nl_array_t* a) {
	assert(a && a->data);
	assert(a->col * a->row <= a->size);
	memset(a->data, 0, a->size * sizeof(float));
}

void nl_array_reshape(nl_array_t* a, int col, int row) {
	assert(a && a->data);
	assert(col * row <= a->size);

	a->col = col;
	a->row = row;
	a->size = col * row;
}

void nl_array_set_val(nl_array_t* a, int r, int c, float v) {
	assert(r < a->row && c < a->col);
	a->data[r*a->row + c] = v;
}

nl_array_t* nl_array_softmax(nl_array_t* d, nl_array_t* s) {
	if (!is_same_shape(d, s)) {
		printf("array softmax failed, dst dim(%d,%d) and src shape(%d,%d)\n",
			d->col, d->row, s->col, s->row);
		abort();
	}

	int size = s->size;
	float max_val = s->data[0];
	for (int i = 0; i < size; i++) {
		if (max_val < s->data[i])
			max_val = s->data[i];
	}

	float sum = 0.0f;
	for (int i = 0; i < size; i++) {
		d->data[i] = expf(s->data[i] - max_val);
		sum += d->data[i];
	}

	for (int i = 0; i < size; i++) {
		d->data[i] /= sum;
	}

	return d;
}

nl_array_t* nl_array_relu( nl_array_t* s) {
	int i;
	#pragma omp parallel for
	for (i = 0; i < s->size; i++) {
		if (s->data[i] < 0)
			s->data[i] = 0;
	}
	return s;
}

nl_array_t* nl_array_relu_grad(nl_array_t* in, nl_array_t* delta) {
	if (!is_same_shape(in, delta)) {
		printf("array relu grad failed, dst dim(%d,%d) and src shape(%d,%d)\n",
			delta->col, delta->row, in->col, in->row);
		abort();
	}

	int i;
	#pragma omp parallel for
	for (i = 0; i < in->size; i++) {
		if (in->data[i] <= 0)
			delta->data[i] = 0;
	}

	return delta;
}

void nl_array_log(nl_array_t* a) {
	printf("[");
	for (int r = 0; r < a->row; r++) {
		printf("[");
		for (int c = 0; c < a->col; c++){
			printf("%g ", a->data[r * a->col + c]);
		}
		printf("]");
		if (r + 1 != a->row)
			printf("\n");
		
	}
	printf("]\n");
}

static float conv_dot(const float* filter, int col, int row, const float* in, int stride) {
	float s = 0;
	const float* data = in;
	const float* kernel = filter;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			s += data[j] * (*kernel);
			kernel++;
		}
		data += stride;
	}

	return s;
}

nl_array_t* nl_array_conv(nl_array_t* d, nl_array_t* s, nl_array_t* filter, float b) {
	int i;
	#pragma omp parallel for private(i)	
	for (i = 0; i < d->row; i++) {
		const float* pos = s->data + i * s->col; //line start pos.
		float* out = d->data + i * d->col;
		for (int j = 0; j < d->col; j++) {
			out[j] = conv_dot(filter->data, filter->col, filter->row, pos + j, s->col) + b;
		}
	}

	return d;
}

static void conv_grad_dot(float* image, float v, nl_array_t* filter, int stride) {
	float* kernel = filter->data;
	const float* data = image;
	for (int i = 0; i < filter->row; i++) {
		for (int j = 0; j < filter->col; j++) {
			*kernel += data[j] * v;
			kernel++;
		}
		data += stride;
	}
}

nl_array_t* nl_array_conv_grad(nl_array_t* filter_delta, nl_array_t* delta_output, nl_array_t* image) {
	nl_array_zero(filter_delta);

	int i;
	#pragma omp parallel for private(i)	
	for (i = 0; i < delta_output->row; i++) {
		float* pos = image->data + i * image->col; //line start pos.
		float* delta = delta_output->data + i * delta_output->col;
		for (int j = 0; j < delta_output->col; j++) {
			conv_grad_dot(pos + j, delta[j], filter_delta, image->col);
		}
	}

	return filter_delta;
}


static float pooling_max(float* in, int pooling, int stride) {
	float* ptr = in;
	float max_v = ptr[0];
	for (int i = 0; i < pooling; i++) {
		for (int j = 0; j < pooling; j++) {
			if (max_v < ptr[j])
				max_v = ptr[j];
		}
		ptr += stride;
	}
	return max_v;
}

nl_array_t* nl_array_pooling(nl_array_t* d, nl_array_t* s, int pooling) {
	int h;
	#pragma omp parallel for private(h)	
	for (h = 0; h < d->row; h++) {
		float* ptr = d->data + h * d->col;
		float* line = s->data + h * s->col * pooling;
		for (int w = 0; w < d->col; w++) {
			ptr[w] = pooling_max(line, pooling, s->col);
			line += pooling;
		}
	}

	return d;
}

static void max_fill_conv_image(float* out, int pooling, int stride, float v) {
	float* ptr = out;

	float* max_pos = ptr;
	float max_v = ptr[0];

	for (int i = 0; i < pooling; i++) {
		for (int j = 0; j < pooling; j++) {
			if (max_v < ptr[j]) {
				max_v = ptr[j];
				max_pos = ptr + j;
			}
			ptr[j] = 0;
		}
		ptr += stride;
	}

	*max_pos = v;
}

nl_array_t* nl_array_pooling_grad(nl_array_t* d, nl_array_t* s, int pooling) {
	int h;
	#pragma omp parallel for private(h)	
	for (h = 0; h < s->row; h++) {
		float *ptr = s->data + h * s->col;
		float *line = d->data + h * d->col * pooling;
		for (int w = 0; w < s->col; w++) {
			max_fill_conv_image(line, pooling, d->col, ptr[w]);
			line += pooling;
		}
	}

	/*set zero to col blank*/
	for (int j = s->col * pooling; j < d->col; j++){
		for (int i = 0; i < s->row * pooling; i++) {
			d->data[i * d->col + j] = 0;
		}
	}

	/*set zero to end of dst array*/
	int pooling_size = s->row * pooling * d->col;
	memset(d->data + pooling_size, 0, (d->size - pooling_size) * sizeof(float));

	return d;
}
