#ifndef __nl_cnn_h_
#define __nl_cnn_h_

#include "nl_mnist.h"

//conv layer param
typedef struct cnn_layer_param {
	//input image param
	int image_w;
	int image_h;

	//filter param
	int filters_n;		//filters num
	int filter_w;		
	int filter_h;
	//out conv size=[w = image_w-filter_w+1, h = image_h - filter_h +1]*filters_n

	//max pool param
	int pooling;

	//full connection layer input size = [w = conv_size_w / pooling, h = conv_size_w / pooling]*filters_n
	int n_full_in; 
	//softmax layer input size;
	int n_softmax_in;
	//output size
	int n_output;
}cnn_layer_param_t;

typedef struct nl_cnn nl_cnn_t;

//create cnn(a conv layer, a full connection layer, a softmax layer)
nl_cnn_t* create_cnn(cnn_layer_param_t* c);
void destroy_cnn(nl_cnn_t* nn);

void cnn_training(nl_cnn_t* nn, nl_data_t* training, int batch_size, float eta);
int cnn_evaluate(nl_cnn_t* nn, nl_data_t* test);

void cnn_log(nl_cnn_t* nn);

#endif
