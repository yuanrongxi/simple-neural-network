#ifndef __nl_ann_h_
#define __nl_ann_h_

#include "nl_mnist.h"

typedef struct nl_ann nl_ann_t;

nl_ann_t* create_ann(const int sizes[], int n);
void destroy_ann(nl_ann_t* nn);

void ann_training(nl_ann_t* nn, nl_data_t* training, int batch_size, float eta);
int ann_evaluate(nl_ann_t* nn, nl_data_t* test);

void ann_log(nl_ann_t* nn);

#endif
