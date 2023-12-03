#ifndef __nl_mnist_h_
#define __nl_mnist_h_

#include "nl_common.h"
#include "nl_array.h"

//mnist files, http://yann.lecun.com/exdb/mnist/
#ifdef WIN32
#define TRAIN_IMAGE ".\\data\\train-images.idx3-ubyte"
#define TRAIN_LABEL ".\\data\\train-labels.idx1-ubyte"
#define TEST_IMAGE ".\\data\\t10k-images.idx3-ubyte"
#define TEST_LABEL ".\\data\\t10k-labels.idx1-ubyte"
#else
#define TRAIN_IMAGE "./data/train-images.idx3-ubyte"
#define TRAIN_LABEL "./data/train-labels.idx1-ubyte"
#define TEST_IMAGE "./data/t10k-images.idx3-ubyte"
#define TEST_LABEL "./data/t10k-labels.idx1-ubyte"
#endif

/*0, 1, 2, 3, 4, 5, 6, 7, 8, 9*/
#define NUMBER_COUNT 10

typedef struct {
	nl_array_t image;			//pic pixel, float 
	nl_array_t* result;			//result array, row=10, col=1
	uint8_t label;				//label value
} train_data_t;

typedef struct {
	int n;
	train_data_t* set;
	float* buff;
}nl_data_t;


void nl_mnist_load(nl_data_t* training_data, nl_data_t* test_data);
void nl_mnist_free(nl_data_t* training_data, nl_data_t* test_data);

void nl_mnist_random_shuffle(nl_data_t* d);

/*test mnist pic*/
void nl_mnist_gen_gpm();

#endif



