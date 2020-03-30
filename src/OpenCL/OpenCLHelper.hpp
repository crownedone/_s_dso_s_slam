#pragma once

#include <string>
#include <opencv2/core/ocl.hpp>
#include "Logging.hpp"

// Simple 'run a kernel' abstraction
bool RunKernel(const std::string& name, const cv::ocl::ProgramSource& KSOURCE,
               std::initializer_list<cv::ocl::KernelArg> args, std::array<size_t, 3> workSize, int nDim = 2, bool sync = false)
{
    // Evtl use build opts (like use_fast_math or smth)
    cv::ocl::Kernel k(name.c_str(), KSOURCE, "");

    if (k.empty())
    {
        LOG_ERROR("Kernel build error %s", name.c_str());
        return false;
    }

    // set kernel Arguments:
    for (int i = 0; i < args.size(); ++i)

    {
        if (k.set(i, args.begin()[i]) != i + 1)
        {
            LOG_ERROR("Failed to set Kernel Argument %d for kernel: %s.", i, name.c_str());
            return false;
        }
    }

    return k.run(nDim, workSize.data(), NULL, sync, cv::ocl::Queue::getDefault());
}
