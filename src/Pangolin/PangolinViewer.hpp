#pragma once
#include "StopWatch.hpp"

#include <pangolin/pangolin.h>

#include "IOWrapper/Output3D.hpp"
#include <map>
#include <deque>
#include <opencv2/core.hpp>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

namespace Viewer
{

class KeyFrameDisplay;

struct GraphConnection
{
    KeyFrameDisplay* from;
    KeyFrameDisplay* to;
    int fwdMarg, bwdMarg, fwdAct, bwdAct;
};


class PangolinViewer : public Output3D
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PangolinViewer(int w, int h);
    virtual ~PangolinViewer();

    /// Start using a custom thread
    inline void start()
    {
        runThread = std::thread(&PangolinViewer::run, this);
    }

    void run();
    void close();

    void addImageToDisplay(std::string name, cv::Mat image);
    void clearAllImagesToDisplay();


    // ==================== Output3DWrapper Functionality ======================
    virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i>& connectivity) override;
    virtual void publishKeyframes(const std::map<int, KeyFrameView>& frames, bool final) override;
    virtual void publishCamPose(const KeyFrameView& kf) override;


    virtual void pushLiveFrame(cv::Mat left) override;
    virtual void pushORBFrame(cv::Mat left) override;
    virtual void pushStereoLiveFrame(cv::Mat left, cv::Mat right) override;
    virtual void pushDepthImage(cv::Mat image) override;
    virtual bool needPushDepthImage() override;

    virtual void join() override;

    virtual void reset() override;
private:

    bool needReset;
    void reset_internal();
    void drawConstraints();

    std::thread runThread;
    bool running;
    int w, h;

    // images rendering
    std::mutex openImagesMutex;
    cv::Mat internalVideoImg;
    cv::Mat internalVideoImg_Right;
    cv::Mat internalORBImg;
    cv::Mat internalKFImg;
    bool videoImgChanged, kfImgChanged, orbImgChanged;



    // 3D model rendering
    std::mutex model3DMutex;
    KeyFrameDisplay* currentCam;
    std::vector<KeyFrameDisplay*> keyframes;
    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> allFramePoses;
    std::map<int, KeyFrameDisplay*> keyframesByKFID;
    std::vector<GraphConnection, Eigen::aligned_allocator<GraphConnection>> connections;



    // render settings
    bool settings_showKFCameras;
    bool settings_showCurrentCamera;
    bool settings_showTrajectory;
    bool settings_showFullTrajectory;
    bool settings_showActiveConstraints;
    bool settings_showAllConstraints;

    float settings_scaledVarTH;
    float settings_absVarTH;
    int settings_pointCloudMode;
    float settings_minRelBS;
    int settings_sparsity;

    // timings
    StopWatch last_track;
    StopWatch last_map;


    std::deque<float> lastNTrackingMs;
    std::deque<float> lastNMappingMs;
};



}
