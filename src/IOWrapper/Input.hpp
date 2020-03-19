#pragma once

#include <string>
#include <memory>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <boost/signals2.hpp>

namespace IO
{

struct FramePack
{
    FramePack() : timestamp(0.0), exposure(1.f), timestamp_slave1(0.0), exposure_slave1(1.f), id(0) {};

    cv::Mat frame;
    cv::Mat frame_slave1;
    cv::Mat depthFrame;

    double timestamp;
    double timestamp_slave1;
    float exposure;
    float exposure_slave1;
    int id;
};

class Input
{
public:
    // make sure to clean up thread
    virtual ~Input();

    /// Signal to receive incoming frames.
    boost::signals2::signal<void(std::shared_ptr<const FramePack>)> onFrame;

    /// playback, notify onFrame for each new frame.
    /// @param waitFrameTimes wait between calls of nextFrame (for the time between two frames)
    void playback(bool waitFrameTimes = true);
private:
    // internally read the next frame.
    virtual std::shared_ptr<const FramePack> nextFrame() = 0;

    // asynchronous
    std::thread m_PlaybackThread;
    bool m_Running;
};


class RGBD_TUM : public Input
{
public:
    // make sure upon destruction that the thread stops first!
    virtual ~RGBD_TUM()
    {
        Input::~Input();
    };

    bool open(const std::string& associationsFilename,
              const std::string& sequenceFolder,
              bool looping = false);

    virtual std::shared_ptr<const FramePack> nextFrame() override;
private:
    std::string associationsFilename;
    std::string sequenceFolder;
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<std::string> vstrImageFilenamesD;
    std::vector<double> vTimestamps;

    // frame count
    size_t count;
    size_t maxCnt;
    bool loop;
};

class Mono_TUM : public Input
{
public:
    // make sure upon destruction that the thread stops first!
    virtual ~Mono_TUM()
    {
        Input::~Input();
    };

    bool open(const std::string& folder,
              bool looping = false);

    virtual std::shared_ptr<const FramePack> nextFrame() override;

private:
    std::vector<std::string> files;
    std::vector<std::string> files1;
    std::vector<double> timestamps;
    std::vector<double> timestamps1;
    std::vector<float> exposures;
    std::vector<float> exposures1;

    std::string path;
    std::string images;
    std::string images1; // optional stereo frames
    std::string calibfile;
    std::string calibfile1;
    std::string gamma;
    std::string vignette;

    bool loadTimestamps();

    // frame count
    size_t count;
    size_t maxCnt;
    bool loop;
};

}