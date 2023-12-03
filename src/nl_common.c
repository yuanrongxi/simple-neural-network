#include "nl_common.h"


int64_t get_sys_time() {
#ifdef WIN32
	FILETIME ft;
	int64_t t;
	GetSystemTimeAsFileTime(&ft);
	t = (int64_t)ft.dwHighDateTime << 32 | ft.dwLowDateTime;
	return t / 10 - 11644473600000000; /* Jan 1, 1601 */
#else 
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_usec + (int64_t)tv.tv_sec * 1000 * 1000;
#endif
}