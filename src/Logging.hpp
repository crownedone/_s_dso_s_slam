#pragma once
#include <memory>

#ifdef CHECK
    #undef CHECK
#endif
#include <glog/logging.h>
#ifdef CHECK
    #undef CHECK
#endif

#include <cstdarg>


static const inline std::string CFormat(const char* format, ...)
{
    // format message
    va_list arglist;
    va_start(arglist, format);

    const int messageSize = 2048;
    std::string message;
    message.resize(messageSize);

#if defined(_MSC_VER)
    auto char_written = _vsnprintf_s_l(&message[0], messageSize, _TRUNCATE, format,
                                       NULL, arglist);
#else
    auto char_written = vsnLOG_INFO(&message[0], messageSize, format, arglist);
#endif

    if (char_written >= 0)
    {
        message.resize(char_written);
        va_end(arglist);
        return message;
    }

    return "";
}

/// Log a message.
/// @param f A printf format string.
/// @param ... Format arguments.
/// @ingroup CommonGroup

#define LOG_INFO(f, ...) VLOG(0) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_IF(cond, f, ...) VLOG_IF(0, cond) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_N(n, f, ...) VLOG_EVERY_N(0, n) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_IF_N(cond, n, f, ...) VLOG_IF_EVERY_N(0, cond, n) << CFormat(f, ##__VA_ARGS__)

#define LOG_INFO_0(f, ...) VLOG(0) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_1(f, ...) VLOG(1) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_2(f, ...) VLOG(2) << CFormat(f, ##__VA_ARGS__)

#define LOG_INFO_0_IF(cond, f, ...) VLOG_IF(0, cond) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_1_IF(cond, f, ...) VLOG_IF(1, cond) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_2_IF(cond, f, ...) VLOG_IF(2, cond) << CFormat(f, ##__VA_ARGS__)

#define LOG_INFO_0_N(n, f, ...) VLOG_EVERY_N(0, n) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_1_N(n, f, ...) VLOG_EVERY_N(1, n) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_2_N(n, f, ...) VLOG_EVERY_N(2, n) << CFormat(f, ##__VA_ARGS__)

#define LOG_INFO_0_IF_N(cond, n, f, ...) VLOG_IF_EVERY_N(0, cond, n) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_1_IF_N(cond, n, f, ...) VLOG_IF_EVERY_N(1, cond, n) << CFormat(f, ##__VA_ARGS__)
#define LOG_INFO_2_IF_N(cond, n, f, ...) VLOG_IF_EVERY_N(2, cond, n) << CFormat(f, ##__VA_ARGS__)

#define LOG_WARNING(f, ...) LOG(WARNING) << CFormat(f, ##__VA_ARGS__)
#define LOG_WARNING_IF(cond, f, ...) LOG_IF(WARNING, cond) << CFormat(f, ##__VA_ARGS__)
#define LOG_WARNING_N(n, f, ...) LOG_EVERY_N(WARNING, n) << CFormat(f, ##__VA_ARGS__)
#define LOG_WARNING_IF_N(cond, n, f, ...) LOG_IF_ERERY_N(WARNING, cond, n) << CFormat(f, ##__VA_ARGS__)

#define LOG_ERROR(f, ...) LOG(ERROR) << CFormat(f, ##__VA_ARGS__)
#define LOG_ERROR_IF(cond, f, ...) LOG_IF(ERROR, cond) << CFormat(f, ##__VA_ARGS__)
#define LOG_ERROR_N(n, f, ...) LOG_EVERY_N(ERROR, n) << CFormat(f, ##__VA_ARGS__)
#define LOG_ERROR_IF_N(cond, n, f, ...) LOG_IF_ERERY_N(ERROR, cond, n) << CFormat(f, ##__VA_ARGS__)

#define LOG_FATAL(f, ...) LOG(FATAL) << CFormat(f, ##__VA_ARGS__)
#define LOG_FATAL_IF(cond, f, ...) LOG_IF(cond, FATAL) << CFormat(f, ##__VA_ARGS__)

/// Debug Logs (disappear in Release builds)
#define DLOG_INFO(f, ...) DLOG(INFO) << CFormat(f, ##__VA_ARGS__)
#define DLOG_INFO_IF(cond, f, ...) DLOG_IF(INFO, cond) << CFormat(f, ##__VA_ARGS__)
#define DLOG_INFO_N(n, f, ...) DLOG_EVERY_N(INFO, n) << CFormat(f, ##__VA_ARGS__)
#define DLOG_INFO_IF_N(cond, n, f, ...) DLOG_IF_EVERY_N(INFO, cond, n) << CFormat(f, ##__VA_ARGS__)

#define DLOG_WARNING(f, ...) DLOG(WARNING) << CFormat(f, ##__VA_ARGS__)
#define DLOG_WARNING_IF(cond, f, ...) DLOG_IF(WARNING, cond) << CFormat(f, ##__VA_ARGS__)
#define DLOG_WARNING_N(n, f, ...) DLOG_EVERY_N(WARNING, n) << CFormat(f, ##__VA_ARGS__)
#define DLOG_WARNING_IF_N(cond, n, f, ...) DLOG_IF_EVERY_N(WARNING, cond, n) << CFormat(f, ##__VA_ARGS__)

#define DLOG_ERROR(f, ...) DLOG(ERROR) << CFormat(f, ##__VA_ARGS__)
#define DLOG_ERROR_IF(cond, f, ...) DLOG_IF(ERROR, cond) << CFormat(f, ##__VA_ARGS__)
#define DLOG_ERROR_N(n, f, ...) DLOG_EVERY_N(ERROR, n) << CFormat(f, ##__VA_ARGS__)
#define DLOG_ERROR_IF_N(cond, n, f, ...) DLOG_IF_EVERY_N(ERROR, cond, n) << CFormat(f, ##__VA_ARGS__)

#define DLOG_FATAL(f, ...) DLOG(FATAL) << CFormat(f, ##__VA_ARGS__)
#define DLOG_FATAL_IF(cond, f, ...) DLOG_IF(FATAL, cond) << CFormat(f, ##__VA_ARGS__)
#define DLOG_FATAL_N(n, f, ...) DLOG_EVERY_N(FATAL, n) << CFormat(f, ##__VA_ARGS__)
#define DLOG_FATAL_IF_N(cond, n, f, ...) DLOG_IF_EVERY_N(FATAL, cond, n) << CFormat(f, ##__VA_ARGS__)

#if !defined(DOXYGEN_DOCU)

#ifndef QUOTE
    #define QUOTE(str) #str
    #define EXPAND_AND_QUOTE(str) QUOTE(str)

    #define __LOC_FIXME__ __FILE__ "(" EXPAND_AND_QUOTE( __LINE__ )") : FIXME : "
    #define __LOC_TODO__ __FILE__ "(" EXPAND_AND_QUOTE( __LINE__ )") : TODO : "
#endif

/// Check if exp == true else return.
/// @param exp Expression that must be true to pass else return; is executed.
/// @note A debug message is logged if returned.
#define CHECK_ARG(exp)                                                                  \
    do                                                                                              \
    {                                                                                       \
        if(!(exp))                                                                                  \
        {                                                                           \
            std::string strExp = EXPAND_AND_QUOTE(exp); \
            if(strExp.find("%") != std::string::npos) \
            { \
                LOG_WARNING("Do not use modulo operator in CHECK_ARG macros!"); \
                return; \
            } \
            LOG_WARNING( "Wrong argument " EXPAND_AND_QUOTE(exp) "."); \
            return;\
        }                                                                           \
    } while(false);

/// Check if exp == true else return with ret.
/// @param exp Expression that must be true to pass else return (ret); is executed.
/// @param ret Return expression that if exp is not true will be returned.
/// @note A debug message is logged if returned.
#define CHECK_ARG_WITH_RET(exp, ret)                                                    \
    do                                                                                              \
    {                                                                                       \
        if(!(exp))                                                                                  \
        {                                                                           \
            std::string strExp = EXPAND_AND_QUOTE(exp); \
            if(strExp.find("%") != std::string::npos) \
            { \
                LOG_WARNING("Do not use modulo operator in CHECK_ARG_WITH_RET macros!"); \
                return ret; \
            } \
            LOG_WARNING( "Wrong argument " EXPAND_AND_QUOTE(exp) ".");            \
            return ret;                                                                             \
        }                                                                           \
    } while(false);

#endif // QUOTE

#define CHECK_ARG_FATAL(exp)                                                                  \
    do                                                                                              \
    {                                                                                       \
        if(!(exp))                                                                                  \
        {                                                                           \
            std::string strExp = EXPAND_AND_QUOTE(exp); \
            if(strExp.find("%") != std::string::npos) \
            { \
                LOG_WARNING("Do not use modulo operator in CHECK_ARG macros!"); \
                return; \
            } \
            LOG_WARNING( "Wrong argument " EXPAND_AND_QUOTE(exp) "."); \
            exit(1);\
        }                                                                           \
    } while(false);