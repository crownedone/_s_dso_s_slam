
#include "PangolinViewer.hpp"
#include "KeyFrameDisplay.hpp"
#include "Logging.hpp"

#include <opencv2/imgproc.hpp>
#include <unordered_set>

// DSO settings
#include "util/settings.hpp"

namespace Viewer
{



PangolinViewer::PangolinViewer(int w, int h)
{
    this->w = w;
    this->h = h;
    running = true;


    {
        std::unique_lock<std::mutex> lk(openImagesMutex);
        internalVideoImg = cv::Mat(h, w, CV_8UC3, cv::Scalar::all(0));
        internalVideoImg_Right = cv::Mat(h, w, CV_8UC3, cv::Scalar::all(0));
        internalKFImg = cv::Mat(h, w, CV_8UC3, cv::Scalar::all(0));
        internalORBImg = cv::Mat(h, w, CV_8UC3, cv::Scalar::all(0));
        videoImgChanged = kfImgChanged = orbImgChanged = true;
    }

    currentCam = new KeyFrameDisplay();
    needReset = false;
}


PangolinViewer::~PangolinViewer()
{
    close();
    runThread.join();
}


void PangolinViewer::run()
{
    LOG_INFO("START PANGOLIN!\n");

    pangolin::CreateWindowAndBind("Main", 2 * w, 2 * h);
    const int UI_WIDTH = 180;

    glEnable(GL_DEPTH_TEST);

    // 3D visualization
    pangolin::OpenGlRenderState Visualization3D_camera(
        pangolin::ProjectionMatrix(w, h, 400, 400, w / 2, h / 2, 0.1, 1000),
        pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY)
    );

    pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
                                              .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w / (float)h)
                                              .SetHandler(new pangolin::Handler3D(Visualization3D_camera));


    // 4 images
    pangolin::View& d_kfDepth = pangolin::Display("imgKFDepth")
                                .SetAspect(w / (float)h);

    pangolin::View& d_video_Right = pangolin::Display("imgKFDepth_Right")
                                    .SetAspect(w / (float)h);
    pangolin::View& d_video = pangolin::Display("imgVideo")
                              .SetAspect(w / (float)h);

    pangolin::View& d_orb = pangolin::Display("imgORB")
                            .SetAspect(w / (float)h);

    pangolin::GlTexture texKFDepth(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    pangolin::GlTexture texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    pangolin::GlTexture texVideo_Right(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    pangolin::GlTexture texORB(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);


    pangolin::CreateDisplay()
    .SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 1.0)
    .SetLayout(pangolin::LayoutEqual)
    .AddDisplay(d_kfDepth)
    .AddDisplay(d_video)
    .AddDisplay(d_video_Right)
    .AddDisplay(d_orb);

    // parameter reconfigure gui
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

    pangolin::Var<int> settings_pointCloudMode("ui.PC_mode", 1, 1, 4, false);

    pangolin::Var<bool> settings_showKFCameras("ui.KFCam", false, true);
    pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam", true, true);
    pangolin::Var<bool> settings_showTrajectory("ui.Trajectory", true, true);
    pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory", false, true);
    pangolin::Var<bool> settings_showActiveConstraints("ui.ActiveConst", true, true);
    pangolin::Var<bool> settings_showAllConstraints("ui.AllConst", false, true);


    pangolin::Var<bool> settings_show3D("ui.show3D", true, true);
    pangolin::Var<bool> settings_showLiveDepth("ui.showDepth", true, true);
    pangolin::Var<bool> settings_showLiveVideo("ui.showVideo", true, true);
    pangolin::Var<bool> settings_showLiveResidual("ui.showResidual", true, true);

    pangolin::Var<bool> settings_showFramesWindow("ui.showFramesWindow", false, true);
    pangolin::Var<bool> settings_showFullTracking("ui.showFullTracking", false, true);
    pangolin::Var<bool> settings_showCoarseTracking("ui.showCoarseTracking", false, true);


    pangolin::Var<int> settings_sparsity("ui.sparsity", 1, 1, 20, false);
    pangolin::Var<double> settings_scaledVarTH("ui.relVarTH", 0.001, 1e-10, 1e10, true);
    pangolin::Var<double> settings_absVarTH("ui.absVarTH", 0.001, 1e-10, 1e10, true);
    pangolin::Var<double> settings_minRelBS("ui.minRelativeBS", 0.1, 0, 1, false);


    pangolin::Var<bool> settings_resetButton("ui.Reset", false, false);


    pangolin::Var<int> settings_nPts("ui.activePoints", dso::setting_desiredPointDensity, 50, 5000, false);
    pangolin::Var<int> settings_nCandidates("ui.pointCandidates", dso::setting_desiredImmatureDensity, 50,
                                            5000, false);
    pangolin::Var<int> settings_nMaxFrames("ui.maxFrames", dso::setting_maxFrames, 4, 10, false);
    pangolin::Var<double> settings_kfFrequency("ui.kfFrequency", dso::setting_kfGlobalWeight, 0.1, 3, false);
    pangolin::Var<double> settings_gradHistAdd("ui.minGradAdd", dso::setting_minGradHistAdd, 0, 15, false);

    pangolin::Var<double> settings_trackFps("ui.Track fps", 0, 0, 0, false);
    pangolin::Var<double> settings_mapFps("ui.KF fps", 0, 0, 0, false);


    // Default hooks for exiting (Esc) and fullscreen (tab).
    while( !pangolin::ShouldQuit() && running )
    {
        // Clear entire screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if(dso::setting_render_display3D)
        {
            // Activate efficiently by object
            Visualization3D_display.Activate(Visualization3D_camera);
            std::unique_lock<std::mutex> lk3d(model3DMutex);
            //pangolin::glDrawColouredCube();
            int refreshed = 0;

            for(KeyFrameDisplay* fh : keyframes)
            {
                float blue[3] = {0, 0, 1};

                if(this->settings_showKFCameras)
                {
                    fh->drawCam(1, blue, 0.1);
                }


                refreshed += (int)(fh->refreshPC(refreshed < 10, this->settings_scaledVarTH,
                                                 this->settings_absVarTH,
                                                 this->settings_pointCloudMode, this->settings_minRelBS, this->settings_sparsity));
                fh->drawPC(1);
            }

            if(this->settings_showCurrentCamera)
            {
                currentCam->drawCam(2, 0, 0.2);
            }

            drawConstraints();
            lk3d.unlock();
        }



        openImagesMutex.lock();

        if (videoImgChanged)
        {
            texVideo.Upload(reinterpret_cast<const void*>(internalVideoImg.data), GL_BGR, GL_UNSIGNED_BYTE);
            texVideo_Right.Upload(reinterpret_cast<const void*>(internalVideoImg_Right.data), GL_BGR, GL_UNSIGNED_BYTE);
        }

        if(kfImgChanged)
        {
            texKFDepth.Upload(reinterpret_cast<const void*>(internalKFImg.data), GL_BGR,
                              GL_UNSIGNED_BYTE);
        }

        if(orbImgChanged)
        {
            texORB.Upload(reinterpret_cast<const void*>(internalORBImg.data), GL_BGR,
                          GL_UNSIGNED_BYTE);
        }

        videoImgChanged = kfImgChanged = orbImgChanged = false;
        openImagesMutex.unlock();




        // update fps counters
        {
            openImagesMutex.lock();
            float sd = 0;

            for(float d : lastNMappingMs)
            {
                sd += d;
            }

            settings_mapFps = lastNMappingMs.size() * 1000.0f / sd;
            openImagesMutex.unlock();
        }
        {
            model3DMutex.lock();
            float sd = 0;

            for(float d : lastNTrackingMs)
            {
                sd += d;
            }

            settings_trackFps = lastNTrackingMs.size() * 1000.0f / sd;
            model3DMutex.unlock();
        }


        if(dso::setting_render_displayVideo)
        {
            d_video.Activate();
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            texVideo.RenderToViewportFlipY();


            d_video_Right.Activate();
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            texVideo_Right.RenderToViewportFlipY();
        }

        if(dso::setting_render_displayDepth)
        {
            d_kfDepth.Activate();
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            texKFDepth.RenderToViewportFlipY();
        }

        if (true)// dso::setting_render_displayResidual)
        {
            d_orb.Activate();
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            texORB.RenderToViewportFlipY();
        }


        // update parameters
        this->settings_pointCloudMode = settings_pointCloudMode.Get();

        this->settings_showActiveConstraints = settings_showActiveConstraints.Get();
        this->settings_showAllConstraints = settings_showAllConstraints.Get();
        this->settings_showCurrentCamera = settings_showCurrentCamera.Get();
        this->settings_showKFCameras = settings_showKFCameras.Get();
        this->settings_showTrajectory = settings_showTrajectory.Get();
        this->settings_showFullTrajectory = settings_showFullTrajectory.Get();

        dso::setting_render_display3D = settings_show3D.Get();
        dso::setting_render_displayDepth = settings_showLiveDepth.Get();
        dso::setting_render_displayVideo =  settings_showLiveVideo.Get();
        dso::setting_render_displayResidual = settings_showLiveResidual.Get();

        dso::setting_render_renderWindowFrames = settings_showFramesWindow.Get();
        dso::setting_render_plotTrackingFull = settings_showFullTracking.Get();
        dso::setting_render_displayCoarseTrackingFull = settings_showCoarseTracking.Get();


        this->settings_absVarTH = settings_absVarTH.Get();
        this->settings_scaledVarTH = settings_scaledVarTH.Get();
        this->settings_minRelBS = settings_minRelBS.Get();
        this->settings_sparsity = settings_sparsity.Get();

        dso::setting_desiredPointDensity = settings_nPts.Get();
        dso::setting_desiredImmatureDensity = settings_nCandidates.Get();
        dso::setting_maxFrames = settings_nMaxFrames.Get();
        dso::setting_kfGlobalWeight = settings_kfFrequency.Get();
        dso::setting_minGradHistAdd = settings_gradHistAdd.Get();


        if(settings_resetButton.Get())
        {
            LOG_INFO("RESET!\n");
            settings_resetButton.Reset();
            dso::setting_fullResetRequested = true;
        }

        // Swap frames and Process Events
        pangolin::FinishFrame();


        if(needReset)
        {
            reset_internal();
        }
    }


    LOG_INFO("QUIT Pangolin thread!\n");
    LOG_INFO("I'll just kill the whole process.\nSo Long, and Thanks for All the Fish!\n");

    exit(1);
}


void PangolinViewer::close()
{
    running = false;
}

void PangolinViewer::join()
{
    runThread.join();
    LOG_INFO("JOINED Pangolin thread!\n");
}

void PangolinViewer::reset()
{
    needReset = true;
}

void PangolinViewer::reset_internal()
{
    model3DMutex.lock();

    for(size_t i = 0; i < keyframes.size(); i++)
    {
        delete keyframes[i];
    }

    keyframes.clear();
    allFramePoses.clear();
    keyframesByKFID.clear();
    connections.clear();
    model3DMutex.unlock();


    openImagesMutex.lock();
    internalVideoImg.setTo(cv::Scalar::all(0));
    internalVideoImg_Right.setTo(cv::Scalar::all(0));
    internalKFImg.setTo(cv::Scalar::all(0));
    internalORBImg.setTo(cv::Scalar::all(0));
    videoImgChanged = kfImgChanged = orbImgChanged = true;
    openImagesMutex.unlock();

    needReset = false;
}


void PangolinViewer::drawConstraints()
{
    if(settings_showAllConstraints)
    {
        // draw constraints
        glLineWidth(1);
        glBegin(GL_LINES);

        glColor3f(0, 1, 0);
        glBegin(GL_LINES);

        for(unsigned int i = 0; i < connections.size(); i++)
        {
            if(connections[i].to == 0 || connections[i].from == 0)
            {
                continue;
            }

            int nAct = connections[i].bwdAct + connections[i].fwdAct;
            int nMarg = connections[i].bwdMarg + connections[i].fwdMarg;

            if(nAct == 0 && nMarg > 0  )
            {
                Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
                glVertex3f((GLfloat) t[0], (GLfloat) t[1], (GLfloat) t[2]);
                t = connections[i].to->camToWorld.translation().cast<float>();
                glVertex3f((GLfloat) t[0], (GLfloat) t[1], (GLfloat) t[2]);
            }
        }

        glEnd();
    }

    if(settings_showActiveConstraints)
    {
        glLineWidth(3);
        glColor3f(0, 0, 1);
        glBegin(GL_LINES);

        for(unsigned int i = 0; i < connections.size(); i++)
        {
            if(connections[i].to == 0 || connections[i].from == 0)
            {
                continue;
            }

            int nAct = connections[i].bwdAct + connections[i].fwdAct;

            if(nAct > 0)
            {
                Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
                glVertex3f((GLfloat) t[0], (GLfloat) t[1], (GLfloat) t[2]);
                t = connections[i].to->camToWorld.translation().cast<float>();
                glVertex3f((GLfloat) t[0], (GLfloat) t[1], (GLfloat) t[2]);
            }
        }

        glEnd();
    }

    if(settings_showTrajectory)
    {
        float colorRed[3] = {1, 0, 0};
        glColor3f(colorRed[0], colorRed[1], colorRed[2]);
        glLineWidth(3);

        glBegin(GL_LINE_STRIP);

        for(unsigned int i = 0; i < keyframes.size(); i++)
        {
            glVertex3f((float)keyframes[i]->camToWorld.translation()[0],
                       (float)keyframes[i]->camToWorld.translation()[1],
                       (float)keyframes[i]->camToWorld.translation()[2]);
        }

        glEnd();
    }

    if(settings_showFullTrajectory)
    {
        float colorGreen[3] = {0, 1, 0};
        glColor3f(colorGreen[0], colorGreen[1], colorGreen[2]);
        glLineWidth(3);

        glBegin(GL_LINE_STRIP);

        for(unsigned int i = 0; i < allFramePoses.size(); i++)
        {
            glVertex3f((float)allFramePoses[i][0],
                       (float)allFramePoses[i][1],
                       (float)allFramePoses[i][2]);
        }

        glEnd();
    }
}






void PangolinViewer::publishGraph(const std::map<uint64_t, Eigen::Vector2i>& connectivity)
{
    if(!dso::setting_render_display3D)
    {
        return;
    }

    if(dso::disableAllDisplay)
    {
        return;
    }

    model3DMutex.lock();
    connections.resize(connectivity.size() / 2);
    int runningID = 0;
    int totalActFwd = 0, totalActBwd = 0, totalMargFwd = 0, totalMargBwd = 0;

    for(std::pair<uint64_t, Eigen::Vector2i> p : connectivity)
    {
        int host = (int)(p.first >> 32);
        int target = (int)(p.first & (uint64_t)0xFFFFFFFF);


        assert(host >= 0 && target >= 0);

        if(host == target)
        {
            assert(p.second[0] == 0 && p.second[1] == 0);
            continue;
        }

        if(host > target)
        {
            continue;
        }

        connections[runningID].from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
        connections[runningID].to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
        connections[runningID].fwdAct = p.second[0];
        connections[runningID].fwdMarg = p.second[1];
        totalActFwd += p.second[0];
        totalMargFwd += p.second[1];

        uint64_t inverseKey = (((uint64_t)target) << 32) + ((uint64_t)host);

        if (inverseKey < connectivity.size())
        {
            Eigen::Vector2i st = connectivity.at(inverseKey);
            connections[runningID].bwdAct = st[0];
            connections[runningID].bwdMarg = st[1];
            totalActBwd += st[0];
            totalMargBwd += st[1];

            runningID++;

        }
    }

    connections.resize(runningID);
    model3DMutex.unlock();
}
void PangolinViewer::publishKeyframes(
    const std::map<int, KeyFrameView>& frames,
    bool final)
{
    if(!dso::setting_render_display3D)
    {
        return;
    }

    if(dso::disableAllDisplay)
    {
        return;
    }

    std::unique_lock<std::mutex> lk(model3DMutex);

    for(auto& kf : frames)
    {
        if(keyframesByKFID.find(kf.first) == keyframesByKFID.end())
        {
            KeyFrameDisplay* kfd = new KeyFrameDisplay();
            keyframesByKFID[kf.first] = kfd;
            keyframes.push_back(kfd);
        }

        keyframesByKFID[kf.first]->setFromKF(kf.second);
    }
}
void PangolinViewer::publishCamPose(const KeyFrameView& kf)
{
    if(!dso::setting_render_display3D)
    {
        return;
    }

    if(dso::disableAllDisplay)
    {
        return;
    }

    std::unique_lock<std::mutex> lk(model3DMutex);

    lastNTrackingMs.push_back(last_track.restart());

    if(lastNTrackingMs.size() > 10)
    {
        lastNTrackingMs.pop_front();
    }

    if(!dso::setting_render_display3D)
    {
        return;
    }

    currentCam->setFromF(kf);
    allFramePoses.push_back(kf.camToWorld.translation().cast<float>());
}

void PangolinViewer::pushORBFrame(cv::Mat left)
{
    std::unique_lock<std::mutex> lk(openImagesMutex);

    if (internalORBImg.size() != left.size())
    {
        cv::resize(left, internalORBImg, internalORBImg.size());
    }
    else
    {
        left.copyTo(internalORBImg);
        //internalORBImg = left;
    }

    orbImgChanged = true;
}

void PangolinViewer::pushLiveFrame(cv::Mat left)
{
    if (!dso::setting_render_displayVideo)
    {
        return;
    }

    if (dso::disableAllDisplay)
    {
        return;
    }

    std::unique_lock<std::mutex> lk(openImagesMutex);

    if (left.type() == CV_32FC3)
    {
        auto ptr = left.ptr<Eigen::Vector3f>();

        for (int i = 0; i < w * h; i++)
        {
            reinterpret_cast<cv::Vec3b*>(internalVideoImg.data)[i][0] =
                reinterpret_cast<cv::Vec3b*>(internalVideoImg.data)[i][1] =
                    reinterpret_cast<cv::Vec3b*>(internalVideoImg.data)[i][2] =
                        ptr[i][0] * 0.8 > 255.0f ? 255.f :
                        ptr[i][0] * 0.8f;

            //internalVideoImg_Right.setTo(cv::Scalar::all(255));
        }
    }
    else
    {
        cv::Mat ll;
        left.convertTo(ll, CV_8U, 0.8);
        cv::cvtColor(ll, internalVideoImg, cv::ColorConversionCodes::COLOR_GRAY2BGR);
        //internalVideoImg_Right.setTo(cv::Scalar::all(255));
    }

    videoImgChanged = true;
}

void PangolinViewer::pushStereoLiveFrame(cv::Mat left, cv::Mat right)
{
    if (!dso::setting_render_displayVideo)
    {
        return;
    }

    if (dso::disableAllDisplay)
    {
        return;
    }

    std::unique_lock<std::mutex> lk(openImagesMutex);

    if (left.type() == CV_32FC3)
    {
        auto ptrL = left.ptr<Eigen::Vector3f>();
        auto ptrR = right.ptr<Eigen::Vector3f>();

        for (int i = 0; i < w * h; i++)
        {
            reinterpret_cast<cv::Vec3b*>(internalVideoImg.data)[i][0] =
                reinterpret_cast<cv::Vec3b*>(internalVideoImg.data)[i][1] =
                    reinterpret_cast<cv::Vec3b*>(internalVideoImg.data)[i][2] =
                        ptrL[i][0] * 0.8 > 255.0f ? 255.f :
                        ptrL[i][0] * 0.8f;

            reinterpret_cast<cv::Vec3b*>(internalVideoImg_Right.data)[i][0] =
                reinterpret_cast<cv::Vec3b*>(internalVideoImg_Right.data)[i][1] =
                    reinterpret_cast<cv::Vec3b*>(internalVideoImg_Right.data)[i][2] =
                        ptrR[i][0] * 0.8 > 255.0f ? 255.f :
                        ptrR[i][0] * 0.8f;
        }
    }
    else
    {
        cv::Mat ll, rr;
        left.convertTo(ll, CV_8U, 0.8);
        right.convertTo(rr, CV_8U, 0.8);
        cv::cvtColor(ll, internalVideoImg, cv::ColorConversionCodes::COLOR_GRAY2BGR);
        cv::cvtColor(rr, internalVideoImg_Right, cv::ColorConversionCodes::COLOR_GRAY2BGR);
    }

    videoImgChanged = true;
}

bool PangolinViewer::needPushDepthImage()
{
    return dso::setting_render_displayDepth;
}
void PangolinViewer::pushDepthImage(cv::Mat image)
{

    if(!dso::setting_render_displayDepth)
    {
        return;
    }

    if(dso::disableAllDisplay)
    {
        return;
    }

    std::unique_lock<std::mutex> lk(openImagesMutex);


    lastNMappingMs.push_back(last_map.restart());

    if(lastNMappingMs.size() > 10)
    {
        lastNMappingMs.pop_front();
    }

    internalKFImg = image;
    kfImgChanged = true;
}


}
