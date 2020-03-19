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


#include "util/Input.hpp"
#include "util/Undistort.hpp"

#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include <gflags/gflags.h>
#include <Logging.hpp>

#include "IOWrapper/Output3DWrapper.hpp"
#include "IOWrapper/ImageDisplay.hpp"


#include <boost/thread.hpp>
#include "util/settings.hpp"
#include "util/globalFuncs.hpp"

#include "util/globalCalib.hpp"

#include "util/NumType.hpp"
#include "DSO_system/FullSystem.hpp"
#include "OptimizationBackend/MatrixAccumulators.hpp"
#include "DSO_system/PixelSelector.hpp"

// Read input
#include "util/DatasetReader.hpp"

#include "IOWrapper/Pangolin/PangolinDSOViewer.hpp"
#include "IOWrapper/SampleOutputWrapper.hpp"

#include "IOWrapper/Input.hpp"

// Path to the sequence folder.
DEFINE_string(sequenceFolder, "", "path to your sequence Folder");
DEFINE_bool(runQuiet, true, "Disable debug output");
DEFINE_bool(useSampleOutput, false, "Disable debug output");
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

int main( int argc, char** argv )
{
    // Init google logging
    // http://rpg.ifi.uzh.ch/docs/glog.html
    google::InitGoogleLogging(argv[0]);

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


    setting_debugout_runquiet = FLAGS_runQuiet;
    usO = FLAGS_useSampleOutput;

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

    int lstart = start;
    int lend = end;

    FullSystem* fullSystem = new FullSystem();
    fullSystem->setGammaFunction(photometricGamma);
    fullSystem->linearizeOperation = (playbackSpeed == 0);

    IOWrap::PangolinDSOViewer* viewer = 0;

    if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }

    if(usO)
    {
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());
    }

    int id = 0;
    input.onFrame.connect([ =, &id, &fullSystem](std::shared_ptr<const IO::FramePack> frame)
    {
        if (fullSystem->initFailed || setting_fullResetRequested)
        {
            if (id < 250 || setting_fullResetRequested)
            {
                LOG_WARNING("RESETTING!\n");

                std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                delete fullSystem;

                for (IOWrap::Output3DWrapper* ow : wraps)
                {
                    ow->reset();
                }

                fullSystem = new FullSystem();
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

        ImageAndExposure* img = undistort->undistort<unsigned char>(
                                    frame->frame, frame->exposure, frame->timestamp);

        ImageAndExposure* img1 = (!frame->frame_slave1.empty()) ?  undistort->undistort<unsigned char>(
                                     frame->frame_slave1, frame->exposure_slave1, frame->timestamp_slave1) : nullptr;

        if (false)// FLAGS_stereomatch)
        {
            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

            //cv::Mat idepthMap(img->h, img_right->w, CV_32FC3, cv::Scalar(0, 0, 0));
            //cv::Mat& idepth_temp = idepthMap;
            //fullSystem->stereoMatch(img_left, img_right, i, idepth_temp);

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            double ttStereoMatch = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
            LOG_INFO(" casting time %f s", ttStereoMatch);
        }
        else
        {
            if (img1)
            {
                fullSystem->addActiveFrame(img, img1, frame->id);
            }
            else
            {
                fullSystem->addActiveFrame(img, frame->id);
            }
        }

        delete img;

        if (img1)
        {
            delete img1;
        }
    });

    input.playback();



    if(viewer != 0)
    {
        viewer->run();
    }

    fullSystem->blockUntilMappingIsFinished();
    fullSystem->printResult("result.txt");
    fullSystem->printFrameLifetimes();

    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }

    LOG_INFO("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    LOG_INFO("DELETE READER!\n");
    delete undistort;

    LOG_INFO("EXIT NOW!\n");
    return 0;
}
