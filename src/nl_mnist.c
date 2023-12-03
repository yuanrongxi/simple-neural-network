/*download mnist file: http://yann.lecun.com/exdb/mnist/ */

#include "nl_mnist.h"
#include "nl_array.h"

typedef struct {
	uint8_t* image;				//image set
	uint8_t* label;				//label set

	size_t n;					//number of items
	int image_row;
	int image_col;
}nl_data_set_t;

static void nl_free_dataset(nl_data_set_t* data_set) {
	if (data_set->image != NULL) {
		free(data_set->image);
		data_set->image = NULL;
	}

	if (data_set->label != NULL) {
		free(data_set->label);
		data_set->label = NULL;
	}

	data_set->image_col = data_set->image_row = 0;
	data_set->n = 0;
}

static uint16_t read_uint16(FILE* f) {
	uint16_t val;
	fread(&val, 1, 2, f);
	return (val << 8) + (val >> 8);
}

static uint32_t read_uint32(FILE* f) {
	uint32_t val;
	fread(&val, 1, 4, f);
	return ((val & 0xff) << 24) + ((val &0xff00) << 8)
		+ ((val & 0x00ff0000) >> 8) + ((val & 0xff000000) >> 24);
}

static bool nl_read_label(const char* label, nl_data_set_t* data_set) {
	FILE* f = fopen(label, "rb");
	if (f == NULL) {
		printf("open label file failed, file: %s\n", label == NULL ? "null" : label);
		return false;
	}

	/*read label file.*/
	uint32_t magic = read_uint32(f);
	if (magic != 2049) {
		printf("invalid label magic:%d(2049)\n", magic);
		return false;
	}

	/*read label data set*/
	uint32_t number = read_uint32(f);
	data_set->label = (uint8_t*)malloc(number);
	data_set->n = number;
	if (fread(data_set->label, 1, data_set->n, f) != data_set->n) {
		printf("invalid labels number %u\n", number);
		return false;
	}

	fclose(f);

	return true;
}

static bool nl_read_image(const char* image, nl_data_set_t* data_set) {
	FILE* f = fopen(image, "rb");
	if (f == NULL) {
		printf("open image file failed, file: %s\n",image == NULL ? "null" : image);
		return false;
	}

	/*read label file.*/
	uint32_t magic = read_uint32(f);
	if (magic != 2051) {
		printf("invalid image magic:%d(2051)\n", magic);
		return false;
	}

	/*read image header*/
	uint32_t number = read_uint32(f);
	if (data_set->n != 0 && data_set->n != number) {
		printf("image number(%u) != label number(%lu)\n", number, data_set->n);
		return false;
	}

	data_set->image_row = read_uint32(f);
	data_set->image_col = read_uint32(f);
	
	size_t sz = number * data_set->image_row * data_set->image_col;
	data_set->image = (uint8_t*)malloc(sz);

	if (fread(data_set->image, 1, sz, f) != sz) {
		printf("invalid labels size: %ux%dx%d\n", number, data_set->image_row, data_set->image_col);
		return false;
	}

	fclose(f);

	return true;
}

static bool nl_read_dataset(const char* image, const char* label, nl_data_set_t* data_set) {

	if (!nl_read_label(label, data_set) || !nl_read_image(image, data_set)) {
		nl_free_dataset(data_set);
		return false;
	}

	return true;
}

static void nl_mnist_load_data(nl_data_set_t* set, nl_data_t* data, bool test){
	int stride = set->image_col * set->image_row;
	data->set = (train_data_t*)malloc(set->n * sizeof(train_data_t));
	data->buff = (float*)malloc(set->n * stride * sizeof(float));
	data->n = set->n;

	uint8_t* image = set->image;
	float* buff = data->buff;

	for (size_t i = 0; i < set->n; i++){
		train_data_t* unit = &data->set[i];
		//set train label val
		unit->label = set->label[i];
		//set train result
		if (!test) {
			unit->result = nl_create_array(1, NUMBER_COUNT);
			unit->result->data[unit->label] = 1.0f;
		}
		else {
			unit->result = NULL;
		}
		//set image array
		nl_set_array(&unit->image, buff, 1, stride);
		for (int k = 0; k < stride; k++)
			*buff++ = (*image++) / 255.0f;
	}
}

static void nl_mnist_release_data(nl_data_t* data) {
	for (int i = 0; i < data->n; i++)
		nl_free_array(data->set[i].result);

	free(data->buff);
	free(data->set);
	data->buff = NULL;
	data->set = NULL;
}

static inline void nl_swap_set(train_data_t* d1, train_data_t* d2) {
	nl_array_t* r;
	if (d1 == d2)
		return;

	r = d2->result;
	d2->result = d1->result;
	d1->result = r;

	uint8_t tmp_label = d1->label;
	d1->label = d2->label;
	d2->label = tmp_label;

	float* tmp = d1->image.data;
	d1->image.data = d2->image.data;
	d2->image.data = tmp;
}

void nl_mnist_load(nl_data_t* training_data, nl_data_t* test_data) {
	nl_data_set_t train_set, test_set;

	if (!nl_read_dataset(TRAIN_IMAGE, TRAIN_LABEL, &train_set)) {
		abort();
	}

	if (!nl_read_dataset(TEST_IMAGE, TEST_LABEL, &test_set)) {
		abort();
	}

	nl_mnist_load_data(&train_set, training_data, false);
	nl_mnist_load_data(&test_set, test_data, true);

	nl_free_dataset(&train_set);
	nl_free_dataset(&test_set);
}

void nl_mnist_free(nl_data_t* training_data, nl_data_t* test_data) {
	assert(training_data && training_data);

	nl_mnist_release_data(training_data);
	nl_mnist_release_data(test_data);
}

void nl_mnist_random_shuffle(nl_data_t* d) {
	for (int i = 0; i < d->n; i++){
		nl_swap_set(d->set + i, d->set + (rand() % d->n));
	}
}

//test interface
static void gen_pgm(nl_data_set_t* set) {
	char buff[100];
	int size = 0, index = rand() % set->n;

	FILE* f = fopen("..\\1.pgm", "wb");

	int stride = set->image_col * set->image_row;
	size = sprintf(buff, "P5\n%d %d\n255\n", set->image_row, set->image_col);
	fwrite(buff, sizeof(char), size, f);

	fwrite(set->image + index * stride, 1, stride, f);
	fclose(f);

	printf("..\\1.pgm is %d\n", set->label[index]);
}

void nl_mnist_gen_gpm() {
	nl_data_set_t set = { NULL, NULL, 0, 0, 0 };
	nl_read_dataset(TRAIN_IMAGE, TRAIN_LABEL, &set);

	printf("train set number = %lu, image row = %d, image col = %d\n", set.n, set.image_row, set.image_row);

	gen_pgm(&set);

	nl_free_dataset(&set);
}
