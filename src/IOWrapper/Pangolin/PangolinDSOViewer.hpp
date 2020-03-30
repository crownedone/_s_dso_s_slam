/**
    This file is part of DSO.

    Copyright 2016 Technical University of Munich and Intel.
    Developed by Jakob Engel <engelj at in dot tum dot de>,
    for more information see <http://vision.in.tum.de/dso>.
    If you use this code, please cite the respective publications as
    listed on the above website.

    DSO is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    DSO is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "StopWatch.hpp"

#include <pangolin/pangolin.h>
#include "boost/thread.hpp"

#include "IOWrapper/Output3DWrapper.hpp"
#include <map>
#include <deque>
#include <opencv2/core.hpp>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
//#include <stdint.h> // portable: uint64_t   MSVC: __int64


namespace dso
{

struct FrameHessian;
struct CalibHessian;
class FrameShell;


namespace IOWrap
{

class KeyFrameDisplay;

struct GraphConnection
{
    KeyFrameDisplay* from;
    KeyFrameDisplay* to;
    int fwdMarg, bwdMarg, fwdAct, bwdAct;
};


class PangolinDSOViewer : public Output3DWrapper
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    PangolinDSOViewer(int w, int h, bool startRunThread = true);
    virtual ~PangolinDSOViewer();

    void run();
    void close();

    void addImageToDisplay(std::string name, const cv::Mat& image);
    void clearAllImagesToDisplay();


    // ==================== Output3DWrapper Functionality ======================
    virtual void publishGraph(const std::map<uint64_t, Eigen::Vector2i>& connectivity) override;
    virtual void publishKeyframes(const std::vector<std::shared_ptr<FrameHessian>>& frames, bool final,
                                  CalibHessian* HCalib) override;
    virtual void publishCamPose(std::shared_ptr<FrameShell> frame, CalibHessian* HCalib) override;


    virtual void pushLiveFrame(std::shared_ptr<FrameHessian> image) override;
    virtual void pushStereoLiveFrame(std::shared_ptr<FrameHessian> image, std::shared_ptr<FrameHessian> image_right) override;
    virtual void pushDepthImage(const cv::Mat& image) override;
    virtual bool needPushDepthImage() override;

    virtual void join() override;

    virtual void reset() override;
private:

    bool needReset;
    void reset_internal();
    void drawConstraints();

    boost::thread runThread;
    bool running;
    int w, h;



    // images rendering
    boost::mutex openImagesMutex;
    cv::Mat internalVideoImg;
    cv::Mat internalVideoImg_Right;
    cv::Mat internalKFImg;
    cv::Mat internalResImg;
    bool videoImgChanged, kfImgChanged, resImgChanged;



    // 3D model rendering
    boost::mutex model3DMutex;
    KeyFrameDisplay* currentCam;
    std::vector<KeyFrameDisplay*> keyframes;
    std::vector<Vec3f, Eigen::aligned_allocator<Vec3f>> allFramePoses;
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



}
