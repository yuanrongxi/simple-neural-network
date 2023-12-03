#include "nl_cnn.h"
#include "nl_array.h"
#include "nl_guass_rand.h"

typedef struct {
	//input image weight and height
	int src_w;
	int src_h;

	//filters, convolution feature map.
	int filters_n;		
	int filter_w;
	int filter_h;

	//conv image weight and height
	int conv_w;
	int conv_h;

	//max pooling
	int pooling;
	int pooling_w;
	int pooling_h;

	//conv layer's bais and weight
	float* b;
	nl_array_t** w;

	float* nabla_b;
	nl_array_t** nabla_w;
}conv_layer_t;

typedef struct {
	int n_in;
	int n_out;

	//full connetction layer (w,b) param
	nl_array_t* b;
	nl_array_t* w;

	nl_array_t* nabla_b;
	nl_array_t* nabla_w;
}full_connection_layer_t;

typedef full_connection_layer_t softmax_layer_t;

typedef struct {
	float* conv_delta_b;
	nl_array_t** conv_delta_w;

	nl_array_t* full_delta_b;
	nl_array_t* full_delta_w;

	nl_array_t* max_delta_b;
	nl_array_t* max_delta_w;

	/*temp array object*/
	nl_array_t conv_out;
	nl_array_t pooling_out;
	nl_array_t full_input;

	/*output array and buffer*/
	float* conv_output;
	float* pooling_output;

	nl_array_t* full_output;

	nl_array_t* max_output;
	nl_array_t* output;

	/*delta output array for backprop*/
	nl_array_t* db_full;
	nl_array_t* db_pooling;
}cnn_runstate_t;

struct nl_cnn {
	conv_layer_t* conv;
	full_connection_layer_t full_layer;
	softmax_layer_t max_layer;
	
	//run state
	cnn_runstate_t state;
	int total_parameters;
};

#define run_state(nn) (&((nn)->state))
#define nn_conv(nn) ((nn)->conv)
#define nn_full(nn) (&((nn)->full_layer))
#define nn_softmax(nn) (&((nn)->max_layer))


////////////////////////////////////////////////////////////////////////////////////////////////////
static conv_layer_t* alloc_conv_layer(cnn_layer_param_t* c) {
	size_t size = sizeof(conv_layer_t);
	conv_layer_t* conv = (conv_layer_t*)calloc(1, size);
	
	conv->src_w = c->image_w;
	conv->src_h = c->image_h;

	conv->filters_n = c->filters_n;
	conv->filter_w = c->filter_w;
	conv->filter_h = c->filter_h;

	conv->conv_w = c->image_w - c->filter_w + 1;
	conv->conv_h = c->image_h - c->filter_h + 1;

	conv->pooling = c->pooling;
	conv->pooling_w = conv->conv_w / conv->pooling;
	conv->pooling_h = conv->conv_h / conv->pooling;

	conv->b = (float*)malloc(sizeof(float)* c->filters_n);
	conv->w = (nl_array_t**)malloc(sizeof(nl_array_t) * c->filters_n);
	conv->nabla_b = (float*)calloc(1, sizeof(float)* c->filters_n);
	conv->nabla_w = (nl_array_t**)malloc(sizeof(nl_array_t)* c->filters_n);

	//init conv filter(w,b)
	int kernel_size = conv->filter_w * conv->filter_h;
	for (int i = 0; i < c->filters_n; i++) {
		conv->b[i] = nl_guass_rand() / kernel_size;
		conv->w[i] = nl_array_randn(c->filter_w, c->filter_h);
		nl_array_div_val(conv->w[i], (float)kernel_size);

		conv->nabla_w[i] = nl_create_array(c->filter_w, c->filter_h);
	}
	return conv;
}

static void free_conv_layer(nl_cnn_t* nn) {
	conv_layer_t* conv = nn_conv(nn);

	for (int i = 0; i < conv->filters_n; i++) {
		nl_free_array(conv->w[i]);
		nl_free_array(conv->nabla_w[i]);
	}

	free(conv->b);
	free(conv->w);
	free(conv->nabla_b);
	free(conv->nabla_w);

	free(conv);
}

static void alloc_full_layer(cnn_layer_param_t* c, nl_cnn_t* nn) {
	full_connection_layer_t* layer = nn_full(nn);
	conv_layer_t* conv = nn_conv(nn);

	layer->n_in = conv->pooling_w * conv->pooling_h * c->filters_n;
	layer->n_out = c->n_softmax_in;

	layer->b = nl_array_randn(1, layer->n_out);
	layer->w = nl_array_randn(layer->n_in, layer->n_out);

	layer->nabla_b = nl_create_array(layer->b->col, layer->b->row);
	layer->nabla_w = nl_create_array(layer->w->col, layer->w->row);
}

static void free_full_layer(nl_cnn_t* nn){
	full_connection_layer_t* layer = nn_full(nn);
	nl_free_array(layer->b);
	nl_free_array(layer->w);
	nl_free_array(layer->nabla_b);
	nl_free_array(layer->nabla_w);
}

static void alloc_softmax_layer(cnn_layer_param_t* c, nl_cnn_t* nn) {
	softmax_layer_t* layer = nn_softmax(nn);
	layer->n_in = c->n_softmax_in;
	layer->n_out = c->n_output;

	layer->b = nl_array_randn(1, layer->n_out);
	layer->w = nl_array_randn(layer->n_in, layer->n_out);

	layer->nabla_b = nl_create_array(layer->b->col, layer->b->row);
	layer->nabla_w = nl_create_array(layer->w->col, layer->w->row);
}

static void free_softmax_layer(nl_cnn_t* nn) {
	softmax_layer_t* layer = nn_softmax(nn);
	nl_free_array(layer->b);
	nl_free_array(layer->w);
	nl_free_array(layer->nabla_b);
	nl_free_array(layer->nabla_w);
}


//memory for intermediate results of calculations
static void alloc_run_state(cnn_layer_param_t* c, nl_cnn_t* nn) {
	int i;
	cnn_runstate_t * st = run_state(nn);
	conv_layer_t* conv = nn_conv(nn);
	full_connection_layer_t* full_layer = nn_full(nn);
	softmax_layer_t* softmax_layer = nn_softmax(nn);

	st->conv_delta_b = (float*)malloc(sizeof(float) * conv->filters_n);
	
	st->conv_delta_w = (nl_array_t**)malloc(conv->filters_n * sizeof(nl_array_t));
	for (i = 0; i < conv->filters_n; i++) {
		st->conv_delta_w[i] = nl_create_array(conv->filter_w, conv->filter_h);
	}

	st->conv_output = (float*)malloc(sizeof(float)* conv->filters_n * conv->conv_w * conv->conv_h); // n * conv-hight * conv->weight
	st->pooling_output = (float*)malloc(sizeof(float*)* full_layer->n_in); // n *  conv-hight * conv->weight / (pooling * pooling)

	nl_set_array(&st->full_input, st->pooling_output, 1, full_layer->n_in);

	st->full_delta_b = nl_create_array(full_layer->b->col, full_layer->b->row);
	st->full_delta_w = nl_create_array(full_layer->w->col, full_layer->w->row);

	st->full_output = nl_create_array(1, full_layer->n_out);

	st->max_delta_b = nl_create_array(softmax_layer->b->col, softmax_layer->b->row);
	st->max_delta_w = nl_create_array(softmax_layer->w->col, softmax_layer->w->row);

	st->max_output = nl_create_array(1, softmax_layer->n_out);
	st->output = nl_create_array(1, softmax_layer->n_out);

	/*delta output array*/
	st->db_full = nl_create_array(1, full_layer->n_out);
	st->db_pooling = nl_create_array(1, full_layer->n_in);
}

static void free_run_state(nl_cnn_t* nn) {
	int i;
	cnn_runstate_t * st = run_state(nn);

	free(st->conv_output);
	free(st->pooling_output);

	free(st->conv_delta_b);
	for (i = 0; i < nn->conv->filters_n; i++) {
		nl_free_array(st->conv_delta_w[i]);
	}

	free(st->conv_delta_w);

	nl_free_array(st->full_delta_b);
	nl_free_array(st->full_delta_w);

	nl_free_array(st->full_output);

	nl_free_array(st->max_delta_b);
	nl_free_array(st->max_delta_w);

	nl_free_array(st->max_output);
	nl_free_array(st->output);

	nl_free_array(st->db_pooling);
	nl_free_array(st->db_full);
}

nl_cnn_t* create_cnn(cnn_layer_param_t* c) {
	nl_cnn_t* nn = (nl_cnn_t*)calloc(1, sizeof(nl_cnn_t));

	//conv layer
	nn_conv(nn) = alloc_conv_layer(c);
	//full-connection layer
	alloc_full_layer(c, nn);
	//softmax layer 
	alloc_softmax_layer(c, nn);

	//alloc runstate memory
	alloc_run_state(c, nn);

	nn->total_parameters = nn_conv(nn)->filters_n * (1 + nn_conv(nn)->filter_w * nn_conv(nn)->filter_h) 
		+ nn_full(nn)->b->size + nn_full(nn)->w->size 
		+ nn_softmax(nn)->b->size + nn_softmax(nn)->w->size;

	return nn;
}

void destroy_cnn(nl_cnn_t* nn) {
	free_conv_layer(nn);
	free_full_layer(nn);
	free_softmax_layer(nn);

	free_run_state(nn);

	free(nn);
}

static int cnn_convolution(nl_cnn_t* nn, nl_array_t* in) {
	conv_layer_t* layer = nn_conv(nn);
	cnn_runstate_t* st = run_state(nn);
	nl_array_t* out = &st->conv_out;


	if (in->size != layer->src_h * layer->src_w) {
		printf("convolution input data size error, input size:%d, src weight:%d, src height:%d\n",
			in->size, layer->src_w, layer->src_h);
		abort();
	}

	nl_array_reshape(in, layer->src_w, layer->src_h);

	int out_size = layer->conv_h * layer->conv_w;
	nl_set_array(out, st->conv_output, layer->conv_w, layer->conv_h);

	for (int i = 0; i < layer->filters_n; i++) {
		nl_array_conv(out, in, layer->w[i], layer->b[i]);
		out->data += out_size; /*next feature*/
	}

	return out_size * layer->filters_n;
}

static void cnn_pooling(nl_cnn_t* nn, float* in, int in_size) {
	conv_layer_t* conv = nn_conv(nn);
	cnn_runstate_t* st = run_state(nn);

	nl_array_t* pin = &st->conv_out, *pout = &st->pooling_out;

	int conv_size = conv->conv_h * conv->conv_w;
	if (in_size != conv_size * conv->filters_n) {
		printf("max pooling input data size error, input size:%d, src weight:%d, src height:%d\n",
			in_size, conv->conv_w, conv->conv_h);
		abort();
	}

	int out_size = conv->pooling_w * conv->pooling_h;
	assert(nn_full(nn)->n_in == out_size * conv->filters_n);

	nl_set_array(pin, in, conv->conv_w, conv->conv_h);
	nl_set_array(pout, st->pooling_output, conv->pooling_w, conv->pooling_h);

	for (int i = 0; i < conv->filters_n; i++) {
		nl_array_pooling(pout, pin, conv->pooling);
		/*flatten array*/
		pin->data += conv_size;
		pout->data += out_size;
	}
}

static nl_array_t* cnn_pooling_relu(nl_cnn_t* nn) {
	cnn_runstate_t* st = run_state(nn);
	return nl_array_relu(&st->full_input);
}

static nl_array_t* cnn_full_forward(nl_cnn_t* nn, nl_array_t* in) {
	full_connection_layer_t* layer = nn_full(nn);
	cnn_runstate_t* st = run_state(nn);

	nl_array_t* out = st->full_output;
	nl_array_dot(out, layer->w, in);
	nl_array_add_self(out, layer->b);

	return nl_array_sigmoid(out);
}

static nl_array_t* cnn_softmax(nl_cnn_t* nn, nl_array_t* in) {
	full_connection_layer_t* layer = nn_softmax(nn);
	cnn_runstate_t* st = run_state(nn);

	nl_array_t* out = st->max_output;
	nl_array_dot(out, layer->w, in);
	nl_array_add_self(out, layer->b);

	return nl_array_softmax(st->output, out);
}

static nl_array_t* cnn_feedforword(nl_cnn_t* nn, nl_array_t* a) {
	nl_array_t* r;
	int conv_size = cnn_convolution(nn, a);
	cnn_pooling(nn, run_state(nn)->conv_output, conv_size);

	r = cnn_full_forward(nn, cnn_pooling_relu(nn));
	return cnn_softmax(nn, r);
}

static void cnn_add_delta(nl_cnn_t* nn, cnn_runstate_t* st, 
		conv_layer_t* conv, full_connection_layer_t* full_layer, softmax_layer_t* softmax_layer) {
	int i; 

	for (i = 0; i < conv->filters_n; i++) {
		conv->nabla_b[i] += st->conv_delta_b[i];
		nl_array_add_self(conv->nabla_w[i], st->conv_delta_w[i]);
	}

	nl_array_add_self(full_layer->nabla_b, st->full_delta_b);
	nl_array_add_self(full_layer->nabla_w, st->full_delta_w);

	nl_array_add_self(softmax_layer->nabla_b, st->max_delta_b);
	nl_array_add_self(softmax_layer->nabla_w, st->max_delta_w);
}

static void cnn_update_nabla(nl_cnn_t* nn, cnn_runstate_t* st,
	conv_layer_t* conv, full_connection_layer_t* full_layer, softmax_layer_t* max_layer, float f) {
	int i;
	float scale = (1.0f * f) / (conv->conv_h * conv->conv_w);
	/*conv weight and bias*/
	for (i = 0; i < conv->filters_n; i++) {
		conv->b[i] -= conv->nabla_b[i] * scale;
		conv->nabla_b[i] = 0;

		nl_array_merge_delta(conv->w[i], conv->nabla_w[i], scale);
	}

	nl_array_merge_delta(full_layer->b, full_layer->nabla_b, f);
	nl_array_merge_delta(full_layer->w, full_layer->nabla_w, f);

	nl_array_merge_delta(max_layer->b, max_layer->nabla_b, f);
	nl_array_merge_delta(max_layer->w, max_layer->nabla_w, f);
}

/*Derivative of softmax function, https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1 */
static void cnn_softmax_backprop(nl_cnn_t* nn, cnn_runstate_t* st, softmax_layer_t* layer, nl_array_t* expect) {
	nl_array_sub(st->max_delta_b, st->output, expect);
	/*cacl softmax layer's weight*/
	nl_array_second_T_dot(st->max_delta_w, st->max_delta_b, st->full_output);
	/*calc full layer's delta*/
	nl_array_first_T_dot(st->db_full, layer->w, st->max_delta_b);
}

static void cnn_full_backprop(nl_cnn_t* nn, cnn_runstate_t* st, full_connection_layer_t* layer) {
	nl_array_mul(st->full_delta_b, st->db_full, nl_array_prime(st->full_output));
	nl_array_second_T_dot(st->full_delta_w, st->full_delta_b, &st->full_input);

	/*calc conv layer's delta*/
	nl_array_first_T_dot(st->db_pooling, layer->w, st->full_delta_b);
}

static void conv_bias_backprop(float* delta_bias, float* delta, int size, int n) {
	for (int i = 0; i < n; i++) {
		float dbias = 0;
		float* pdb = delta + i * size;
		for (int j = 0; j < size; j++)
			dbias += pdb[j];

		delta_bias[i] = dbias;
	}
}

static void conv_pooling_backprop(nl_cnn_t* nn, cnn_runstate_t* st, conv_layer_t* layer) {
	int delta_size, conv_image_size;
	nl_array_t* conv_image = &st->conv_out, *delta = &st->pooling_out;

	nl_set_array(conv_image, st->conv_output, layer->conv_w, layer->conv_h);
	nl_set_array(delta, st->db_pooling->data, layer->pooling_w, layer->pooling_h);

	conv_image_size = conv_image->size;
	delta_size = delta->size;

	if (delta_size * layer->filters_n != st->db_pooling->size) {
		printf("max pooling out data size error, db pooling size:%d, src weight:%d, src height:%d\n",
			st->db_pooling->size, layer->pooling_w, layer->pooling_h);
		abort();
	}

	for (int i = 0; i < layer->filters_n; i++) {
		nl_array_pooling_grad(conv_image, delta, layer->pooling);
		conv_image->data += conv_image_size;
		delta->data += delta_size;
	}
}

static void conv_backprop(nl_cnn_t* nn, cnn_runstate_t* st, conv_layer_t* layer, nl_array_t* image) {
	int conv_image_size;
	nl_array_t* conv_image = &st->conv_out;

	nl_set_array(conv_image, st->conv_output, layer->conv_w, layer->conv_h);
	conv_image_size = conv_image->size;

	if (image->size != layer->src_h * layer->src_w) {
		printf("conv image data size error, input size:%d, src weight:%d, src height:%d\n",
			image->size, layer->src_w, layer->src_h);
		abort();
	}

	nl_array_reshape(image, layer->src_w, layer->src_h);

	for (int i = 0; i < layer->filters_n; i++) {
		nl_array_conv_grad(st->conv_delta_w[i], conv_image, image);
		conv_image->data += conv_image_size;
	}
}

static void cnn_conv_backprop(nl_cnn_t* nn, cnn_runstate_t* st, conv_layer_t* layer, nl_array_t* image) {
	/*relu backprop*/
	nl_array_relu_grad(&st->full_input, st->db_pooling);
	/*bais backprop*/
	conv_bias_backprop(st->conv_delta_b, st->db_pooling->data, st->db_pooling->size / layer->filters_n, layer->filters_n);
	/*pooling backprop*/
	conv_pooling_backprop(nn, st, layer);
	/*conv backprop*/
	conv_backprop(nn, st, layer, image);
}


static void cnn_update_batch(nl_cnn_t* nn, train_data_t* data, int n, float eta) {
	int i;

	cnn_runstate_t* st = run_state(nn);
	conv_layer_t* conv = nn_conv(nn);
	full_connection_layer_t* full_layer = nn_full(nn);
	softmax_layer_t* softmax_layer = nn_softmax(nn);
	
	nl_array_t* image;

	for (i = 0; i < n; i++) {
		image = &(data[i].image);
		/*forword*/
		cnn_feedforword(nn, image);
		/*backprop*/
		cnn_softmax_backprop(nn, st, softmax_layer, data[i].result);
		cnn_full_backprop(nn, st, full_layer);
		cnn_conv_backprop(nn, st, conv, image);
		/*accumulate delta*/
		cnn_add_delta(nn, st, conv, full_layer, softmax_layer);
	}

	cnn_update_nabla(nn, st, conv, full_layer, softmax_layer, eta / n);
}

void cnn_training(nl_cnn_t* nn, nl_data_t* training, int batch_size, float eta) {
	nl_mnist_random_shuffle(training);

	int remain = training->n;
	train_data_t* data = training->set;
	while (remain > 0) {
		int stride = (batch_size > remain) ? remain : batch_size;
		cnn_update_batch(nn, data, stride, eta);
		data += stride;
		remain -= stride;
	}
}

int cnn_evaluate(nl_cnn_t* nn, nl_data_t* test) {
	nl_array_t* r;
	int count = 0;
	for (int i = 0; i < test->n; i++) {
		r = cnn_feedforword(nn, &(test->set[i].image));
		if (nl_array_argmax(r) == test->set[i].label)
			count++;
	}

	return count;
}

static inline void cnn_conv_log(conv_layer_t* layer) {
	printf("cnn conv layer: \n");
	printf("\tsrc weight:%d, src height:%d, conv weight:%d, conv height:%d\n", layer->src_w, layer->src_h, layer->conv_w, layer->conv_h);
	printf("\tfeature map count:%d, weight:%d, height:%d\n", layer->filters_n, layer->filter_w, layer->filter_h);
	printf("\tpooling:%d, weight:%d, height:%d\n", layer->pooling, layer->pooling_w, layer->pooling_h);
}

static inline void cnn_full_log(full_connection_layer_t* layer) {
	printf("cnn full layer: \n");
	printf("\tin size:%d, out size:%d\n", layer->n_in, layer->n_out);

	printf("\tbais count:%d\n", layer->b->size);
	printf("\tweight count:%d\n", layer->w->size);
}

static inline void cnn_softmax_log(softmax_layer_t* layer) {
	printf("cnn softmax layer: \n");
	printf("\tin size:%d, out size:%d\n", layer->n_in, layer->n_out);

	printf("\tbais count:%d\n", layer->b->size);
	printf("\tweight count:%d\n", layer->w->size);
}

void cnn_log(nl_cnn_t* nn) {
	printf("cnn network:\n");
	cnn_conv_log(nn_conv(nn));
	cnn_full_log(nn_full(nn));
	cnn_softmax_log(nn_softmax(nn));
	printf("cnn total parameters count:%d\n", nn->total_parameters);
}