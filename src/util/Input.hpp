#pragma once

#include <opencv2/core/mat.hpp>
#include <boost/signals2.hpp>
#include <thread>

class Input
{
public:
    boost::signals2::signal<void(const cv::Mat)> onFrame;

    void captureFunction()
    {

        onFrame(cv::Mat());

    };

    void main_fnc()
    {
        //m_Thread = std::thread(captureFunction);
    }

private:
    std::thread m_Thread;
};