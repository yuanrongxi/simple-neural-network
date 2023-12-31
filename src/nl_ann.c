#include "nl_ann.h"
#include "nl_array.h"

typedef struct {
	nl_array_t** signal; //input signal

	nl_array_t** nabla_b;
	nl_array_t** nabla_w;

	nl_array_t** delta_b;
	nl_array_t** delta_w;
}nl_ann_runstate_t;

struct nl_ann {
	int layers;
	int op_count;
	nl_array_t** b;
	nl_array_t** w;

	nl_ann_runstate_t state;
	int total_parameters;
};

#define run_state(nn) (&((nn)->state))
#define run_signal(nn) ((nn)->state.signal)


/* once allocate memory, no need to allocate memory during program execution */
static void alloc_run_state(nl_ann_t* ann) {
	nl_ann_runstate_t* st = run_state(ann);
	
	st->signal = (nl_array_t**)malloc(ann->layers *sizeof(nl_array_t*));

	st->nabla_b = (nl_array_t**)malloc(ann->op_count * sizeof(nl_array_t*));
	st->delta_b = (nl_array_t**)malloc(ann->op_count * sizeof(nl_array_t*));

	st->nabla_w = (nl_array_t**)malloc(ann->op_count * sizeof(nl_array_t*));
	st->delta_w = (nl_array_t**)malloc(ann->op_count * sizeof(nl_array_t*));

	st->signal[0] = NULL; 
	for (int i = 0; i < ann->op_count; i++){
		st->signal[i + 1] = nl_create_array(ann->b[i]->col, ann->b[i]->row);
		st->delta_b[i] = nl_create_array(ann->b[i]->col, ann->b[i]->row);
		st->nabla_b[i] = nl_create_array(ann->b[i]->col, ann->b[i]->row);
		st->delta_w[i] = nl_create_array(ann->w[i]->col, ann->w[i]->row);
		st->nabla_w[i] = nl_create_array(ann->w[i]->col, ann->w[i]->row);
	}
}

static void free_run_state(nl_ann_t* ann) {
	nl_ann_runstate_t* st = run_state(ann);

	for (int i = 0; i < ann->op_count; i++){
		nl_free_array(st->signal[i+1]);
		nl_free_array(st->delta_b[i]);
		nl_free_array(st->nabla_b[i]);
		nl_free_array(st->delta_w[i]);
		nl_free_array(st->nabla_w[i]);
	}

	free(st->signal);
	free(st->delta_b);
	free(st->nabla_b);
	free(st->delta_w);
	free(st->nabla_w);
}

nl_ann_t* create_ann(const int sizes[], int n) {
	int i;
	assert(n > 2);

	nl_ann_t* nn = (nl_ann_t*)malloc(sizeof(nl_ann_t));
	nn->b = (nl_array_t**)malloc((n - 1) * sizeof(nl_array_t*));
	nn->w = (nl_array_t**)malloc((n - 1) * sizeof(nl_array_t*));
	nn->layers = n;
	nn->op_count = n - 1;
	nn->total_parameters = 0;

	/*gen biases*/
	for (i = 0; i < nn->op_count; ++i){
		nn->b[i] = nl_array_randn(1, sizes[i + 1]);
		nn->total_parameters += nn->b[i]->size;
	}

	/*gen weights*/
	for (i = 0; i < nn->op_count; ++i) {
		nn->w[i] = nl_array_randn(sizes[i], sizes[i + 1]);
		nn->total_parameters += nn->w[i]->size;
	}

	/*alloc run state array*/
	alloc_run_state(nn);

	return nn;
}

void destroy_ann(nl_ann_t* nn) {
	assert(nn != NULL);

	free_run_state(nn);

	for (int i = 0; i < nn->op_count; i++){
		nl_free_array(nn->w[i]);
		nl_free_array(nn->b[i]);
	}

	free(nn->b);
	free(nn->w);

	free(nn);
}

static int feedforward(nl_ann_t* nn, nl_array_t* a) {
	int i;
	nl_array_t* y = NULL;
	nl_array_t** signal = run_signal(nn);
	
	signal[0] = a;

	for (i = 0; i < nn->op_count; i++) {
		y = signal[i + 1];
		/*y = w.x+b*/
		nl_array_dot(y, nn->w[i], signal[i]);
		nl_array_add_self(y, nn->b[i]);
		/*sigmod*/
		nl_array_sigmoid(y);
	}

	return nl_array_argmax(y);
}

int ann_evaluate(nl_ann_t* nn, nl_data_t* test) {
	int count = 0;

	for (int i = 0; i < test->n; i++) {
		if (feedforward(nn, &(test->set[i].image)) == test->set[i].label)
			count++;
	}

	return count;
}


static void ann_backprop(nl_ann_t* nn, nl_ann_runstate_t* state, nl_array_t* x, nl_array_t* y) {
	int i;
	nl_array_t* z;
	nl_array_t** signal = run_signal(nn);
	signal[0] = x;

	/*feedbackword*/
	for (i = 0; i < nn->op_count; i++){
		z = signal[i + 1];
		
		/*y = wx+b*/
		nl_array_dot(z, nn->w[i], signal[i]);
		nl_array_add_self(z, nn->b[i]);

		nl_array_sigmoid(z);
	}

	/*calc first delta*/
	int last = nn->op_count - 1;
	nl_array_t* delta = state->delta_b[last];

	nl_array_sub(delta, signal[last+1], y);
	nl_array_mul_self(delta, nl_array_prime(signal[last + 1]));
	
	nl_array_second_T_dot(state->delta_w[last], delta, signal[last]);

	for (i = last - 1; i >= 0; i--) {
		//transpose weight
		delta = state->delta_b[i];

		nl_array_first_T_dot(delta, nn->w[i + 1], state->delta_b[i + 1]);
		nl_array_mul_self(delta, nl_array_prime(signal[i+1]));

		nl_array_second_T_dot(state->delta_w[i], delta, signal[i]);
	}
}

static void ann_update_batch(nl_ann_t* nn, train_data_t* data, int n, float eta) {
	nl_ann_runstate_t* state = run_state(nn);
	int i, k;

	train_data_t* dp = data;
	for (i = 0; i < n; i++) {
		ann_backprop(nn, state, &(dp->image), dp->result);
		dp++;

		for (k = 0; k < nn->op_count; k++){
			nl_array_add_self(state->nabla_b[k], state->delta_b[k]);
			nl_array_add_self(state->nabla_w[k], state->delta_w[k]);
		}
	}

	float f = eta / n;
	for (k = 0; k < nn->op_count; k++){
		nl_array_merge_delta(nn->w[k], state->nabla_w[k], f);
		nl_array_merge_delta(nn->b[k], state->nabla_b[k], f);
	}
}


void ann_training(nl_ann_t* nn, nl_data_t* training, int batch_size, float eta) {
	if (training->n <= 0) {
		assert(false);
		return;
	}
	
	nl_mnist_random_shuffle(training);

	int remain = training->n;
	train_data_t* data = training->set;
	while (remain > 0) {
		int stride = (batch_size > remain) ? remain : batch_size;
		ann_update_batch(nn, data, stride, eta);
		data += stride;
		remain -= stride;
	}
}

void ann_log(nl_ann_t* nn) {
	printf("ann network:\n");
	printf("\tlayers:%d\n", nn->layers);

	int bais = 0, weights = 0;
	for (int i = 0; i < nn->layers - 1; i++) {
		bais += nn->b[i]->size;
		weights += nn->w[i]->size;
	}

	printf("\tbiases:%d\n", bais);
	printf("\tweights:%d\n", weights);
	printf("\ttotal parameters:%d\n", nn->total_parameters);
}