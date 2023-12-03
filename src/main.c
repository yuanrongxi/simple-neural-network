#include <stdio.h>
#include <time.h>

#include "nl_guass_rand.h"
#include "nl_mnist.h"
#include "nl_array.h"
#include "nl_ann.h"
#include "nl_cnn.h"

#include "unit_test.h"

static void run_ann(const int sizes[], int sizes_len, int epochs, int batch_size, float eta) {
	nl_ann_t* nn;
	nl_data_t training, test;

	nl_mnist_load(&training, &test);

	assert(training.n == 60000);
	assert(test.n == 10000);

	nn = create_ann(sizes, sizes_len);
	
	ann_log(nn);

	int64_t start_ts = SYS_MS();
	for (int i = 0; i < epochs; i++){
		int64_t batch_ts = SYS_MS();
		ann_training(nn, &training, batch_size, eta);
		printf("Epoch %d: %d / %d, cost: %.2fs\n", i, ann_evaluate(nn, &test), test.n, (float)(SYS_MS() - batch_ts) / 1000.0);
	}
	printf("all cost: %.2fs\n", (float)(SYS_MS() - start_ts) / 1000.0);
	destroy_ann(nn);
	nl_mnist_free(&training, &test);
}

static void run_cnn(int epochs, int batch_size, float eta) {
	cnn_layer_param_t param = { 28, 28, 20, 5, 5, 2, 10 * 12 * 12, 100, 10 };
	nl_data_t training, test;

	nl_mnist_load(&training, &test);

	assert(training.n == 60000);
	assert(test.n == 10000);

	nl_cnn_t* nn = create_cnn(&param);
	cnn_log(nn);

	//nn_traing_debug(nn, &training, 1, 0.1f);
	int64_t start_ts = SYS_MS();
	for (int i = 0; i < epochs; i++){
		int64_t batch_ts = SYS_MS();
		cnn_training(nn, &training, batch_size, eta);
		printf("cnn Epoch %d: %d / %d, cost: %.2fs\n", i, cnn_evaluate(nn, &test), test.n, (float)(SYS_MS() - batch_ts) / 1000.0);
	}
	printf("all cost: %.2fs\n", (float)(SYS_MS() - start_ts) / 1000.0);

	destroy_cnn(nn);

	nl_mnist_free(&training, &test);
}

static void run_unit_test() {
	nl_mnist_gen_gpm();
	test_array_dot();
	test_array_T_dot();
	test_array_op();
	test_sigmoid();
	test_array_transpose();

	test_array_randn();
	test_random_shuffle();
	test_mnist_load();

	test_soft_max();
	test_relu();
	test_pooling();
	test_pooling_grad();
	test_conv();
	test_conv_grad();

	test_create_cnn();
	test_create_ann();
}

/*execute command: ./nn cnn or ./nn ann*/
int main(int argc, char* argv[])
{
	srand(time(NULL));

	if (argc == 2 && strcmp(argv[1], "cnn") == 0){
		run_cnn(30, 10, 0.1f);
	}
	else {
		int arr[3] = { 784, 30, NUMBER_COUNT };
		run_ann(arr, sizeof(arr) / sizeof(int), 30, 10, 3.0);
	}
	//run_unit_test();

	return 0;
}

