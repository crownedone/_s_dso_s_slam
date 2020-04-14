#pragma once

#include <opencv2/core.hpp>
#include <sophus/se3.hpp>
#include <map>
#include <vector>

namespace Viewer
{
/// Generalized Output Interface handling all the views:
template<int ppp>
struct InputPointSparse
{
    float u;
    float v;
    float idpeth;
    float idepth_hessian;
    float relObsBaseline;
    int numGoodRes;
    unsigned char color[ppp];
    unsigned char status;
};

struct KeyFrameView
{
    int id;
    int w, h;
    float fx, fy, cx, cy;

    Sophus::SE3d PRE_camToWorld;
    Sophus::SE3d camToWorld;

    // 8 = MAX_RES_PER_POINT!
    std::vector<InputPointSparse<8>> pts;
};


// 2D windows (opencv)
void displayImage(const char* windowName, cv::Mat img, bool autoSize = false);
void displayImageStitch(const char* windowName, const std::vector<cv::Mat>& images,
                        int cc = 0, int rc = 0);

int waitKey(int milliseconds);
void closeAllWindows();

// 3D (Pangolin) window
class Output3D
{
public:
    virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i>& connectivity) {};
    virtual void publishKeyframes(const std::map<int, KeyFrameView>& frames, bool final) {};
    virtual void publishCamPose(const KeyFrameView& kf) {};

    virtual void pushLiveFrame(cv::Mat left) {};
    virtual void pushStereoLiveFrame(cv::Mat left, cv::Mat right) {};
    virtual void pushORBFrame(cv::Mat left) {};
    virtual void pushDepthImage(cv::Mat image) {};
    virtual bool needPushDepthImage();

    // For view of idepth.
    virtual void pushDepthImageFloat(cv::Mat image)
    {
        displayImage("depth image", image);
        waitKey(1);
    };

    virtual void reset() {};
    virtual void join() {};
};

}