// From http://www.linuxjournal.com/article/5574
#pragma once

#ifdef _WIN32
    #include <stdint.h>
    #include <sys/timeb.h>
    #include <sys/types.h>
    #include <WinSock2.h>
#else
    #include <chrono>
#endif

/// Linux and Windows compatible Stopwatch class
class StopWatch
{
private:
    // MSVC does not have a high_resolution_clock in chrono until VS2015
#if defined(WIN32)
    /// First time point.
    int64_t mStart;
#else
    /// First time point.
    std::chrono::high_resolution_clock::time_point mStart;
#endif

public:
    /// Start is automatically called.
    StopWatch();

    /// Start stop watch.
    void start();

    /// Stop and return time in milliseconds.
    double stop();

    /// Restarts timer and returns time until now.
    double restart();
};