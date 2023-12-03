#include <math.h>
#include <time.h>

#include "nl_guass_rand.h"

#ifdef WIN32
#define	inline __inline
#endif

typedef struct {
	bool has_gauss;
	double gauss;
}random_state;

static inline double randf() {
	return (rand() / (RAND_MAX + 1.0));
}
static double legacy_gauss(random_state* state) {
	if (state->has_gauss) {
		double ret = state->gauss;
		state->has_gauss = false;
		state->gauss = 0.0f;
		return ret;
	}
	else {
		double f, x1, x2, r2;

		do {
			x1 = 2.0 * randf() - 1.0;
			x2 = 2.0 * randf() - 1.0;
			r2 = x1 * x1 + x2 * x2;
		} while (r2 > 1.0f || r2 == 0.0f);

		/* Polar method, a more efficient version of the Box-Muller approach. */
		f = sqrt(-2.0 * log(r2) / r2);
		/* Keep for next call */
		state->gauss = f * x1;
		state->has_gauss = true;
		return f * x2;
	}
}

static random_state rstate = { false, 0.0f };

float nl_guass_rand() {
	return (float)legacy_gauss(&rstate);
}
