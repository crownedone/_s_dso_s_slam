// From http://www.linuxjournal.com/article/5574
#pragma once

#ifndef _TIMES_H
#define _TIMES_H

#ifdef _WIN32
#include <stdint.h>
#include <sys/timeb.h>
#include <sys/types.h>
#include <WinSock2.h>

//int gettimeofday(struct timeval* t,void* timezone);

#define __need_clock_t
#include <time.h>

// MSVC defines this in winsock2.h!?
/*typedef struct timeval {
	long tv_sec;
	long tv_usec;
} timeval;*/

void gettimeofday(struct timeval* t, void* timezone);
//int gettimeofday(struct timeval * tp, struct timezone * tzp);

/* Structure describing CPU time used by a process and its children.  */
struct tms {
	clock_t tms_utime;          /* User CPU time.  */
	clock_t tms_stime;          /* System CPU time.  */

	clock_t tms_cutime;         /* User CPU time of dead children.  */
	clock_t tms_cstime;         /* System CPU time of dead children.  */
};

/* Store the CPU time used by this process and all its
   dead children (and their dead children) in BUFFER.
   Return the elapsed real time, or (clock_t) -1 for errors.
   All times are in CLK_TCKths of a second.  */
clock_t times (struct tms *__buffer);

typedef long long suseconds_t ;

#endif
#endif