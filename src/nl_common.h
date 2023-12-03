#ifndef __nl_common_h
#define __nl_common_h

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include <omp.h>

#ifndef bool
#define bool uint8_t
#define true 1
#define false 0
#endif

#ifndef inline
#define	inline __inline
#endif

#ifdef WIN32
#pragma warning(disable: 4996) //this is about sprintf, snprintf, vsnprintf
#include <windows.h>
#else
#include <sys/time.h>
#endif


#define nl_max(a, b) ((a) >= (b) ? (a) : (b))
#define nl_min(a, b) ((a) <= (b) ? (a) : (b))

int64_t get_sys_time();
#define SYS_MS() (get_sys_time()/1000)

#endif