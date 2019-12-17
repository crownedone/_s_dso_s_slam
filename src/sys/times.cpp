#include "sys/times.h"

void gettimeofday(struct timeval* t, void* timezone)
{
    struct _timeb timebuffer;
    _ftime(&timebuffer);
    t->tv_sec = static_cast<long>(timebuffer.time);
    t->tv_usec = 1000 * timebuffer.millitm;
}

/*
    int gettimeofday(struct timeval * tp, struct timezone * tzp) {

    static const uint64_t EPOCH = ((uint64_t)116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);
    time = ((uint64_t)file_time.dwLowDateTime);
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec = (long)((time - EPOCH) / 10000000L);
    tp->tv_usec = (long)(system_time.wMilliseconds * 1000);

    return 0;
    }*/

clock_t times (struct tms* __buffer)
{

    __buffer->tms_utime = clock();
    __buffer->tms_stime = 0;
    __buffer->tms_cstime = 0;
    __buffer->tms_cutime = 0;
    return __buffer->tms_utime;
}

#ifdef _WIN32
#define VC_EXTRALEAN
#include "Windows.h"

static LARGE_INTEGER GetQueryPerformanceFrequency()
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return freq;
}

static LARGE_INTEGER FREQUENCY = GetQueryPerformanceFrequency();
#endif

StopWatch::StopWatch()
{
	start();
}

void StopWatch::start()
{
#if defined(_WIN32)
	LARGE_INTEGER l;
	QueryPerformanceCounter(&l);
	mStart = l.QuadPart;
#else
	mStart = std::chrono::high_resolution_clock::now();
#endif
}

double StopWatch::restart()
{
	double milliseconds = 0;

#if defined(_WIN32)
	LARGE_INTEGER now;
	QueryPerformanceCounter(&now);
	milliseconds = static_cast<float>(static_cast<double>(now.QuadPart - mStart) /
		(static_cast<double>(FREQUENCY.QuadPart)));
	milliseconds *= 1000;
	mStart = now.QuadPart;
#else
	auto now = std::chrono::high_resolution_clock::now();
	milliseconds =
		static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(now - mStart)
			.count());
	milliseconds /= 1000;
	mStart = now;
#endif

	return milliseconds;
}

double StopWatch::stop()
{
	auto start = mStart;

	auto milliseconds = restart();

	mStart = start;

	return milliseconds;
}
