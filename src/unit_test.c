#include <stdio.h>

#include "nl_guass_rand.h"
#include "nl_mnist.h"
#include "nl_array.h"
#include "nl_ann.h"
#include "nl_cnn.h"


void test_randn() {
	for (int i = 0; i < 30; i++){
		printf("%f\n", nl_guass_rand());
	}
}

void test_array_dot() {
	nl_array_t* a = nl_create_array(3, 1); //[[1, 2, 3]]
	a->data[0] = 1;
	a->data[1] = 2;
	a->data[2] = 3;
	/*a->data[3] = 4;
	a->data[4] = 5;
	a->data[5] = 6;*/
	printf("a = ");
	nl_array_log(a);
	nl_array_t* b = nl_create_array(1, 3); //[[1], [2], [3]]
	b->data[0] = 1;
	b->data[1] = 2;
	b->data[2] = 3;
	/*b->data[3] = 4;
	b->data[4] = 5;
	b->data[5] = 6;*/
	printf("b = ");
	nl_array_log(b);

	nl_array_t* c = nl_create_array(a->col, b->row);
	nl_array_dot(c, a, b);
	printf("c = ");
	nl_array_log(c);

	nl_free_array(a);
	nl_free_array(b);
	nl_free_array(c);
}

void test_array_T_dot() {
	nl_array_t* a = nl_create_array(2, 3); //[[1, 2, 3]]
	a->data[0] = 1;
	a->data[1] = 2;
	a->data[2] = 3;
	a->data[3] = 4;
	a->data[4] = 5;
	a->data[5] = 6;
	printf("a = ");
	nl_array_log(a);
	nl_array_t* b = nl_create_array(2, 3); //[[1], [2], [3]]
	b->data[0] = 1;
	b->data[1] = 2;
	b->data[2] = 3;
	b->data[3] = 4;
	b->data[4] = 5;
	b->data[5] = 6;
	printf("b = ");
	nl_array_log(b);

	nl_array_t* t = nl_create_array(3, 2);
	nl_array_transpose(t, a);

	nl_array_t* t1 = nl_create_array(3, 2);
	nl_array_transpose(t1, b);

	nl_array_t* c = nl_create_array(t->col, b->row);
	nl_array_dot(c, t, b);
	printf("c = ");
	nl_array_log(c);

	nl_array_first_T_dot(c, a, b);
	printf("c = ");
	nl_array_log(c);

	nl_array_second_T_dot(c, t, t1);
	printf("c = ");
	nl_array_log(c);

	nl_free_array(a);
	nl_free_array(b);
	nl_free_array(c);
	nl_free_array(t);
	nl_free_array(t1);
}

void test_array_op() {
	nl_array_t* a = nl_create_array(3, 2); //[[1, 2, 3]]
	a->data[0] = 1;
	a->data[1] = 2;
	a->data[2] = 3;
	a->data[3] = 4;
	a->data[4] = 5;
	a->data[5] = 6;
	printf("a = \n");
	nl_array_log(a);

	nl_array_add_val(a, 1);
	printf("a + 1 = \n");
	nl_array_log(a);

	nl_array_sub_val(a, 2);
	printf("a - 2 = \n");
	nl_array_log(a);

	nl_array_mul_val(a, 2);
	printf("a * 2 = \n");
	nl_array_log(a);

	nl_array_div_val(a, 2);
	printf("a * 2 = \n");
	nl_array_log(a);

	nl_array_t* b = nl_create_array(3, 2); //[[2, 2, 2]]
	b->data[0] = 2;
	b->data[1] = 2;
	b->data[2] = 2;
	b->data[3] = 2;
	b->data[4] = 2;
	b->data[5] = 2;
	printf("b = \n");
	nl_array_log(b);

	nl_array_t* c = nl_create_array(3, 2);
	nl_array_add(c, a, b);
	printf("a + b=\n");
	nl_array_log(c);

	nl_array_sub(c, a, b);
	printf("a - b=\n");
	nl_array_log(c);

	nl_array_mul(c, a, b);
	printf("a * b=\n");
	nl_array_log(c);


	nl_array_div(c, a, b);
	printf("a / b=\n");
	nl_array_log(c);

	nl_array_sigmoid(a);
	printf("sigmoid a=\n");
	nl_array_log(a);

	nl_array_sigmoid_prime(a);
	printf("sigmoid prime=\n");
	nl_array_log(a);

	nl_free_array(a);
	nl_free_array(b);
	nl_free_array(c);
}

void test_sigmoid() {
	float data[2] = { 0.1f, 0.2f };
	nl_array_t *a, d;
	a = &d;
	nl_set_array(a, data, 2, 1);
	a->data[0] = 0.1f;
	a->data[1] = 0.2f;

	nl_array_sigmoid(a);
	nl_array_log(a);
	printf("\n");
	nl_array_log(nl_array_sigmoid_prime(a));
}

void test_array_transpose() {
	float a_data[6] = { 1, 2, 3, 4, 5, 6 };
	float c_data[6], b_data[6];
	nl_array_t d1, *a, d2, *b, d3, *c;
	a = &d1;
	c = &d2;
	b = &d3;
	nl_set_array(a, a_data, 3, 2);
	nl_set_array(b, b_data, 3, 2);
	nl_set_array(c, c_data, 2, 3);
	nl_array_log(a);

	printf("transpose a=\n");
	nl_array_log(nl_array_transpose(c, a));

	printf("transpose c=\n");
	nl_array_log(nl_array_transpose(b, c));


	nl_array_zero_reshape(a, 1, 3);
	a->data[0] = 1;
	a->data[1] = 2;
	a->data[2] = 3;
	printf("a=\n");
	nl_array_log(a);
	printf("transpose a=\n");
	nl_array_log(nl_array_transpose(c, a));
	nl_array_log(nl_array_transpose(a, c));
}


void test_array_randn() {
	static int sizes[3] = { 4, 5, 2 };
	nl_array_t* biases[2];
	nl_array_t* weights[2];

	//gen b
	for (size_t i = 0; i < sizeof(sizes) / sizeof(int)-1; i++) {
		biases[i] = nl_array_randn(1, sizes[i + 1]);
	}

	//gen w
	for (size_t i = 0; i < sizeof(sizes) / sizeof(int)-1; i++) {
		weights[i] = nl_array_randn(sizes[i], sizes[i + 1]);
	}

	printf("biases:\n[");
	for (int i = 0; i < 2; i++) {
		printf("array:");
		nl_array_log(biases[i]);
	}
	printf("]\n");

	printf("weights:\n[");
	for (int i = 0; i < 2; i++) {
		printf("array:");
		nl_array_log(weights[i]);
	}
	printf("]\n");
}

void test_soft_max() {
	float data[4] = { -1, 0, 3, 5 };
	nl_array_t a, *pa;
	pa = &a;
	nl_set_array(pa, data, 4, 1);

	nl_array_softmax(pa, pa);

	nl_array_log(pa);
}

void test_relu() {
	float data[10] = { -1, 0, -2, 1, 3, 1, -1, 9, 10, 3 };
	nl_array_t a, *pa;
	pa = &a;

	nl_set_array(pa, data, 10, 1);
	nl_array_log(pa);

	nl_array_relu(pa);
	printf("relu:\n");
	nl_array_log(pa);
}

void test_random_shuffle() {
	nl_data_t d;
	d.buff = NULL;
	d.n = 10;
	train_data_t set[10];
	d.set = set;
	float* ptr = NULL;

	printf("src:");
	for (size_t i = 0; i < d.n; i++){
		d.set[i].image.data = ptr + i;
		d.set[i].label = i;
		d.set[i].result = nl_create_array(1, NUMBER_COUNT);
		for (int j = 0; j < NUMBER_COUNT; j++)
			d.set[i].result->data[j] = 0;

		d.set[i].result->data[i % NUMBER_COUNT] = 1.0f;

		printf("%d ", d.set[i].label);
	}
	printf("\nshtffle:");

	nl_mnist_random_shuffle(&d);

	for (size_t i = 0; i < d.n; i++) {
		printf("%d ", d.set[i].label);
		assert(ptr + d.set[i].label == d.set[i].image.data);
		assert(d.set[i].result->data[d.set[i].label % NUMBER_COUNT] == 1.0f);

		nl_free_array(d.set[i].result);
	}
	printf("\n");
}

void test_mnist_load() {
	nl_data_t training, test;
	nl_mnist_load(&training, &test);

	assert(training.n == 60000);
	assert(test.n == 10000);

	nl_array_log(&training.set[training.n - 1].image);

	nl_mnist_random_shuffle(&training);

	nl_mnist_free(&training, &test);
}

void test_create_ann() {
	int sizes[3] = { 4, 5, 2 };
	nl_ann_t* h = create_ann(sizes, 3);

	ann_log(h);

	destroy_ann(h);
}

void test_conv() {
	float d1[16] = {
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
		12, 13, 14, 15
	};

	float d2[4] = {
		1, 1,
		1, 1
	};

	float d3[9] = { 0 };

	nl_array_t a, *pa, b, *pb, c, *pc;
	pa = &a;
	pb = &b;
	pc = &c;

	nl_set_array(pa, d1, 4, 4);
	nl_set_array(pb, d2, 2, 2);
	nl_set_array(pc, d3, 3, 3);

	printf("image:\n");
	nl_array_log(pa);

	printf("filter:\n");
	nl_array_log(pb);

	nl_array_conv(pc, pa, pb, 0);

	printf("conv:\n");
	nl_array_log(pc);
}

void test_conv_grad() {
	float d1[9] = {
		0, 1, 2,
		4, 5, 6,
		8, 9, 10
	};

	float d3[4] = { 0, 0.1f,
		0.1f, 0 };

	float d2[4] = {
		0, 0,
		0, 0
	};

	nl_array_t a, *pa, b, *pb, c, *pc;
	pa = &a;
	pb = &b;
	pc = &c;

	nl_set_array(pa, d1, 3, 3);
	nl_set_array(pb, d2, 2, 2);
	nl_set_array(pc, d3, 2, 2);

	printf("image:\n");
	nl_array_log(pa);

	printf("delta:\n");
	nl_array_log(pc);

	nl_array_conv_grad(pb, pc, pa);

	printf("filter:\n");
	nl_array_log(pb);
}

void test_pooling() {
	float d1[24] = {
		0, 1, 2, 3,
		4, 5, 6, 7,
		8, 9, 10, 11,
		12, 13, 14, 15,
		16, 17, 18, 19,
		20, 21, 22, 23
	};

	float d2[6] = { 0 };
	nl_array_t a, *pa;
	pa = &a;

	nl_set_array(pa, d1, 4, 6);
	nl_array_log(pa);

	nl_array_t b, *pb;
	pb = &b;
	nl_set_array(pb, d2, 2, 3);

	nl_array_pooling(pb, pa, 2);
	printf("max pooling:\n");
	nl_array_log(pb);
}

void test_pooling_grad() {
	float d1[35] = {
		0, 1, 2, 3, 3,
		4, 5, 6, 7, 7,
		8, 9, 10, 11, 11,
		12, 13, 14, 15, 15,
		16, 17, 18, 19, 19,
		20, 21, 22, 23, 23,
		20, 21, 22, 23, 23
	};

	float d2[6] = { 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f };
	nl_array_t a, *pa;
	pa = &a;

	nl_set_array(pa, d1, 5, 7);
	nl_array_log(pa);

	nl_array_t b, *pb;
	pb = &b;
	nl_set_array(pb, d2, 2, 3);

	nl_array_pooling_grad(pa, pb, 2);
	printf("max pooling grad:\n");
	nl_array_log(pa);
}

void test_create_cnn() {
	cnn_layer_param_t param = { 28, 28, 20, 5, 5, 2, 10 * 12 * 12, 100, 10 };

	nl_cnn_t* nn = create_cnn(&param);

	cnn_log(nn);

	destroy_cnn(nn);
}
