#pragma once
#include "Input.hpp"
#include "Logging.hpp"

#include <opencv2/imgcodecs.hpp>
namespace IO
{

Input::~Input()
{
    if (m_PlaybackThread.joinable())
    {
        m_Running = false;
        m_PlaybackThread.join();
    }
};

void Input::playback(bool waitFrameTimes /*= true*/)
{
    if (m_Running)
    {
        m_Running = false;

        if (m_PlaybackThread.joinable())
        {
            m_PlaybackThread.join();
        }
    }

    m_Running = true;
    m_PlaybackThread = std::thread([this, waitFrameTimes]()
    {
        auto frame = nextFrame();

        if (!frame)
        {
            LOG_ERROR("Input not set up. aborting");
            m_Running = false;
            return;
        }

        // expecting unix timestamp (seconds since epoch)
        double lastTimestamp = frame->timestamp;
        onFrame(frame);

        while (m_Running)
        {
            frame = nextFrame();

            if (!frame)
            {
                LOG_ERROR("Input error. aborting");
                m_Running = false;
                return;
            }

            double timestamp = frame->timestamp;

            if (waitFrameTimes)
            {
                double diff = std::abs(timestamp - lastTimestamp);
                unsigned int diffMS = static_cast<unsigned int>(diff * 1000 * 1000);
                std::this_thread::sleep_for(std::chrono::microseconds(diffMS));
            }

            onFrame(frame);

            lastTimestamp = timestamp;
        }

        m_Running = false;
    });
}
}