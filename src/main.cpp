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

#include <ORB_SLAM2_System/System.hpp>
#include "util/Input.hpp"
#include "util/Undistort.hpp"

#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include <gflags/gflags.h>
#include <Logging.hpp>

#include "IOWrapper/Output3D.hpp"

#include <boost/thread.hpp>
#include "util/settings.hpp"
#include "util/globalFuncs.hpp"

#include "util/globalCalib.hpp"

#include "util/NumType.hpp"
#include "DSO_system/FullSystem.hpp"
#include "OptimizationBackend/MatrixAccumulators.hpp"
#include "DSO_system/PixelSelector.hpp"

// Read input
#include <boost/filesystem.hpp>

#include "Pangolin/PangolinViewer.hpp"

#include "IOWrapper/Input.hpp"
#include <opencv2/core/ocl.hpp>

// Path to the sequence folder.
DEFINE_string(sequenceFolder, "", "path to your sequence Folder");
DEFINE_string(orbVocab, "", "ORB-Vocabulary");
DEFINE_string(orbSettings, "", "ORB-Settings yaml");
DEFINE_bool(runQuiet, true, "Disable debug output");
DEFINE_bool(GPU, false, "USE GPU Calculation");

DEFINE_int32(preset, 0,
             "0 - DEFAULT settings : \n"\
             "- %s real-time enforcing\n"\
             "- 2000 active points\n"\
             "- 5-7 active frames\n"\
             "- 1-6 LM iteration each KF\n"\
             "- original image resolution\n"\
             "- no speedup \n\n" \
             "1 - DEFAULT settings : \n"\
             "- %s real-time enforcing\n"\
             "- 2000 active points\n"\
             "- 5-7 active frames\n"\
             "- 1-6 LM iteration each KF\n"\
             "- original image resolution\n"\
             "- 1x speedup \n\n"\
             "2 - FAST settings:\n"\
             "- %s real-time enforcing\n"\
             "- 800 active points\n"\
             "- 4-6 active frames\n"\
             "- 1-4 LM iteration each KF\n"\
             "- 424 x 320 image resolution\n"\
             "- no speedup"\
             "3 - FAST settings:\n"\
             "- %s real-time enforcing\n"\
             "- 800 active points\n"\
             "- 4-6 active frames\n"\
             "- 1-4 LM iteration each KF \n" \
             "- 424 x 320 image resolution \n" \
             "- 5x speedup");

/// Validator accessor to validate input parameter: Settings
static bool ValidateFile(const char* flagname, const std::string& value)
{
    if (!value.empty() || boost::filesystem::exists(value))   // value is ok
    {
        return true;
    }

    LOG_ERROR("Invalid value for --%s: %s\n", flagname, value.c_str());
    exit(1);
    return false;
}
static const bool sequenceFolderValidator = gflags::RegisterFlagValidator(&FLAGS_sequenceFolder,
                                            &ValidateFile);


double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start = 0;
int end = 100000;
bool prefetch = false;
float playbackSpeed =
    0; // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload = false;
bool usO = false;


int mode = 0;

bool firstRosSpin = false;

using namespace dso;

void settingsDefault(int preset)
{
    LOG_INFO("\n=============== PRESET Settings: ===============\n");

    if(preset == 0 || preset == 1)
    {
        LOG_INFO("DEFAULT settings:\n"
                 "- %s real-time enforcing\n"
                 "- 2000 active points\n"
                 "- 5-7 active frames\n"
                 "- 1-6 LM iteration each KF\n"
                 "- original image resolution\n", preset == 0 ? "no " : "1x");

        playbackSpeed = (preset == 0 ? 0.f : 1.f);
        preload = preset == 1;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations = 6;
        setting_minOptIterations = 1;

        // Go back to previous values
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_kfGlobalWeight = 1.0;

        setting_logStuff = true;
    }

    if(preset == 2 || preset == 3)
    {
        LOG_INFO("FAST settings:\n"
                 "- %s real-time enforcing\n"
                 "- 800 active points\n"
                 "- 4-6 active frames\n"
                 "- 1-4 LM iteration each KF\n"
                 "- 424 x 320 image resolution\n", preset == 0 ? "no " : "5x");

        playbackSpeed = (preset == 2 ? 0.f : 5.f);
        preload = preset == 3;
        setting_desiredImmatureDensity = 600;
        setting_desiredPointDensity = 800;
        setting_minFrames = 4;
        setting_maxFrames = 6;
        setting_maxOptIterations = 4;
        setting_minOptIterations = 1;

        benchmarkSetting_width = 424;
        benchmarkSetting_height = 320;

        setting_logStuff = false;
    }

    LOG_INFO("==============================================\n");
}


void initializeOpenCL()
{
    assert(cv::ocl::haveOpenCL());
    auto m_Context = cv::ocl::Context();
    auto m_Queue = cv::ocl::Queue();
    bool hasSVM = cv::ocl::haveSVM();

    try
    {
        // prefer GPU context:
        m_Context.create(cv::ocl::Device::TYPE_ALL);

        if (m_Context.ndevices() <= 0)
        {
            LOG_WARNING("No OpenCL device(s) found");
            m_Context = cv::ocl::Context::getDefault();
            m_Queue = cv::ocl::Queue::getDefault();
        }
        else
        {
            int idx = 0;
            LOG_INFO("Found %d OpenCL Devices:", m_Context.ndevices());

            for (int i = 0; i < m_Context.ndevices(); ++i)
            {
                cv::ocl::Device dev = m_Context.device(i);
                LOG_INFO("(%s, %s, %s)",
                         dev.name().c_str(),
                         dev.OpenCLVersion().c_str(), dev.OpenCL_C_Version().c_str());

                // We prefer intel:
                if (dev.isIntel())
                {
                    idx = i;
                }
            }

            LOG_INFO("Selected: Device(Name, OclVer, OclCVer) = (%s, %s, %s)",
                     m_Context.device(idx).name().c_str(),
                     m_Context.device(idx).OpenCLVersion().c_str(),
                     m_Context.device(idx).OpenCL_C_Version().c_str());
            // Select this device globally as c (Contexts) Default device.
            cv::ocl::Device(m_Context.device(idx));
            m_Queue.cv::ocl::Queue::create(m_Context, m_Context.device(idx));
        }
    }
    catch (std::exception& e)
    {
        LOG_ERROR("OCL init error %s", e.what());
    }

    // Set specific Context and queue as static for the whole environment!
    cv::ocl::Context::getDefault(false) = std::move(m_Context);
    cv::ocl::Queue::getDefault() = std::move(m_Queue);

    //Context = cv::ocl::Context::getDefault(false);
    //Queue = cv::ocl::Queue::getDefault();
    //Context.setUseSVM(hasSVM); // Use SVM if avaialable
};

int main( int argc, char** argv )
{
    // Init google logging
    // http://rpg.ifi.uzh.ch/docs/glog.html
    google::InitGoogleLogging(argv[0]);

    // Seems to have no impact on performance
    Eigen::initParallel();

    // log options (there are more options available)
    FLAGS_alsologtostderr = 1;
    FLAGS_colorlogtostderr = 1;

    LOG_INFO("Starting DSO:");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Only valid with sequence:
    if (FLAGS_sequenceFolder.empty())
    {
        LOG_ERROR("missing argument --sequenceFolder=X");
        exit(1);
    }

    setting_UseOpenCL = FLAGS_GPU;
    setting_debugout_runquiet = FLAGS_runQuiet;

    if (FLAGS_GPU)
    {
        initializeOpenCL();
    }


    settingsDefault(FLAGS_preset);

    IO::Mono_TUM input;
    dso::Undistort* undistort;
    float* photometricGamma = nullptr;


    if (!input.open(FLAGS_sequenceFolder))
    {
        LOG_ERROR("Cannot open!");
        exit(1);
    }
    else
    {
        std::string path = FLAGS_sequenceFolder;
        std::string calibfile = path + "/camera.txt";
        std::string gamma = path + "/pcalib.txt";
        std::string vignette = path + "/vignette.png";
        undistort = Undistort::getUndistorterForFile(calibfile, gamma, vignette);

        if (undistort)
        {
            photometricGamma = undistort->photometricUndist->getG();

            Eigen::Matrix3f K = undistort->getK().cast<float>();
            int w = undistort->getSize()[0];
            int h = undistort->getSize()[1];
            setGlobalCalib(w, h, K);
            // Set baseline for stereo (if available)
            baseline = undistort->getBl();
        }
    }



    if(setting_photometricCalibration > 0 && photometricGamma == 0)
    {
        LOG_ERROR("dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
        exit(1);
    }

    std::shared_ptr<Viewer::PangolinViewer> viewer = nullptr;

    if(!disableAllDisplay)
    {
        viewer = std::make_shared<Viewer::PangolinViewer>(wG[0], hG[0]);
        // viewer->start(); // viewer thread start
    }

    auto orbSystem = std::make_unique<ORB_SLAM2::System>(FLAGS_orbVocab, FLAGS_orbSettings, ORB_SLAM2::System::MONOCULAR, viewer);
    std::unique_ptr<FullSystem> fullSystem = std::make_unique<FullSystem>();
    fullSystem->setGammaFunction(photometricGamma);
    fullSystem->linearizeOperation = (playbackSpeed == 0);

    if(viewer)
    {
        fullSystem->outputWrapper.push_back(viewer);
    }

    int id = 0;

    input.onFrame.connect([ =, &id, &fullSystem, &orbSystem](std::shared_ptr<const IO::FramePack> frame)
    {
        if (fullSystem->initFailed || setting_fullResetRequested)
        {
            if (id < 250 || setting_fullResetRequested)
            {
                LOG_WARNING("RESETTING!\n");

                auto wraps = fullSystem->outputWrapper;
                fullSystem = nullptr;

                for (auto& ow : wraps)
                {
                    ow->reset();
                }

                fullSystem = std::make_unique<FullSystem>();
                fullSystem->setGammaFunction(photometricGamma);
                fullSystem->linearizeOperation = (playbackSpeed == 0);


                fullSystem->outputWrapper = wraps;

                setting_fullResetRequested = false;
            }
        }

        if (fullSystem->isLost)
        {
            LOG_ERROR("Tracking lost!");
            return;
        }

        StopWatch sw;
        auto img = undistort->undistort(
                       frame->frame, frame->exposure, frame->timestamp);

        auto img1 = (!frame->frame_slave1.empty()) ?  undistort->undistort(
                        frame->frame_slave1, frame->exposure_slave1, frame->timestamp_slave1) : nullptr;
        LOG_INFO("Preprocessing time %f ms", sw.restart());

        if (img1)
        {
            if (img->image8u_umat.empty())
            {
                orbSystem->TrackStereo(img->image8u, img1->image8u, frame->timestamp);
            }
            else
            {
                orbSystem->TrackStereo(img1->image8u_umat, img1->image8u_umat, frame->timestamp);
            }

            fullSystem->addActiveFrame(img, img1, frame->id);
        }
        else
        {
            if (img->image8u_umat.empty())
            {
                orbSystem->TrackMonocular(img->image8u, frame->timestamp);
            }
            else
            {
                orbSystem->TrackMonocular(img->image8u_umat, frame->timestamp);
            }


            fullSystem->addActiveFrame(img, frame->id);
        }
    });

    input.playback();



    if(viewer != 0)
    {
        viewer->run();
    }

    orbSystem->Shutdown();

    // Save camera trajectory
    orbSystem->SaveTrajectoryTUM("CameraTrajectory.txt");
    orbSystem->SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    fullSystem->blockUntilMappingIsFinished();
    fullSystem->printResult("dsoResult.txt");
    fullSystem->printFrameLifetimes();

    for(auto ow : fullSystem->outputWrapper)
    {
        ow->join();
        ow = nullptr;
    }

    LOG_INFO("DELETE READER!\n");
    delete undistort;

    LOG_INFO("EXIT NOW!\n");
    return 0;
}
