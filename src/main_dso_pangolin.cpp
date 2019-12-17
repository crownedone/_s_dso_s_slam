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



#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#include <gflags/gflags.h>
#include <sys/Logging.hpp>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"

#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector.h"

// Read input
#include "util/DatasetReader.h"

#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

// Path to the sequence folder.
DEFINE_string(sequenceFolder, "", "path to your sequence Folder");
DEFINE_bool(runQuiet, true, "Disable debug output");
DEFINE_bool(useSampleOutput, false, "Disable debug output");
DEFINE_int32(preset, 0, \
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
             "- 1-4 LM iteration each KF\n"\
             "- 424 x 320 image resolution\n"\
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
static const bool sequenceFolderValidator = gflags::RegisterFlagValidator(&FLAGS_sequenceFolder, &ValidateFile);


double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start = 0;
int end = 100000;
bool prefetch = false;
float playbackSpeed = 0; // 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
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
        setting_desiredImmatureDensity = 1500;
        setting_desiredPointDensity = 2000;
        setting_minFrames = 5;
        setting_maxFrames = 7;
        setting_maxOptIterations = 6;
        setting_minOptIterations = 1;

        setting_logStuff = false;
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
    ImageFolderReader* reader = new ImageFolderReader(FLAGS_sequenceFolder);
    reader->setGlobalCalibration();

    if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
    {
        LOG_ERROR("dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
        exit(1);
    }

    int lstart = start;
    int lend = end;
    int linc = 1;

    if(reverse)
    {
        LOG_INFO("REVERSE!!!!");
        lstart = end - 1;

        if(lstart >= reader->getNumImages())
        {
            lstart = reader->getNumImages() - 1;
        }

        lend = start;
        linc = -1;
    }

    FullSystem* fullSystem = new FullSystem();
    fullSystem->setGammaFunction(reader->getPhotometricGamma());
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

    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]()
    {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;

        for(int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc)
        {
            idsToPlay.push_back(i);

            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis - tsPrev) / playbackSpeed);
            }
        }


        std::vector<ImageAndExposure*> preloadedImages;

        if(preload)
        {
            LOG_INFO("LOADING ALL IMAGES!\n");

            for(int ii = 0; ii < (int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
            }
        }

        struct timeval tv_start;

        gettimeofday(&tv_start, NULL);

        clock_t started = clock();

        double sInitializerOffset = 0;

        StopWatch sw;

        for(int ii = 0; ii < (int)idsToPlay.size(); ii++)
        {
            if(!fullSystem->initialized)    // if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];


            ImageAndExposure* img;

            if(preload)
            {
                img = preloadedImages[ii];
            }
            else
            {
                img = reader->getImage(i);
            }

            bool skipFrame = false;

            if(playbackSpeed != 0)
            {
                struct timeval tv_now;
                gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) + (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));

                if(sSinceStart < timesToPlayAt[ii])
                {
                    std::this_thread::sleep_for(std::chrono::microseconds((int)((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000)));
                }
                else if(sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2))
                {
                    LOG_INFO("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame = true;
                }
            }

            if(!skipFrame)
            {
                fullSystem->addActiveFrame(img, i);
            }

            delete img;

            if(fullSystem->initFailed || setting_fullResetRequested)
            {
                if(ii < 250 || setting_fullResetRequested)
                {
                    LOG_INFO("RESETTING!\n");

                    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                    delete fullSystem;

                    for(IOWrap::Output3DWrapper* ow : wraps)
                    {
                        ow->reset();
                    }

                    fullSystem = new FullSystem();
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed == 0);


                    fullSystem->outputWrapper = wraps;

                    setting_fullResetRequested = false;
                }
            }

            if(fullSystem->isLost)
            {
                LOG_INFO("LOST!!\n");
                break;
            }

            LOG_WARNING("Cycle: %f [ms]", sw.restart());
        }

        fullSystem->blockUntilMappingIsFinished();
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);


        fullSystem->printResult("result.txt");


        int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0]) - reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
        LOG_INFO("\n======================"
                 "\n%d Frames (%.1f fps)"
                 "\n%.2fms per frame (single core); "
                 "\n%.2fms per frame (multi core); "
                 "\n%.3fx (single core); "
                 "\n%.3fx (multi core); "
                 "\n======================\n\n",
                 numFramesProcessed, numFramesProcessed / numSecondsProcessed,
                 MilliSecondsTakenSingle / numFramesProcessed,
                 MilliSecondsTakenMT / (float)numFramesProcessed,
                 1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
                 1000 / (MilliSecondsTakenMT / numSecondsProcessed));

        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f * (ended - started) / (float)(CLOCKS_PER_SEC * reader->getNumImages()) << " "
                  << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) / (float)reader->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }

    });


    if(viewer != 0)
    {
        viewer->run();
    }

    runthread.join();

    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
        delete ow;
    }



    LOG_INFO("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    LOG_INFO("DELETE READER!\n");
    delete reader;

    LOG_INFO("EXIT NOW!\n");
    return 0;
}
