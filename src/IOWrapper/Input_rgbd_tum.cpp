#include "Input.hpp"
#include "Logging.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace std;
namespace IO
{
bool RGBD_TUM::open(const std::string& associationsFilename_, const std::string& sequenceFolder_,
                    bool looping)
{
    loop = looping;

    LOG_INFO("Opening %s", associationsFilename_.c_str());

    CHECK_ARG_WITH_RET(!associationsFilename_.empty() &&
                       boost::filesystem::exists(associationsFilename_) &&
                       boost::filesystem::is_regular_file(associationsFilename_), false);

    CHECK_ARG_WITH_RET(!sequenceFolder_.empty() &&
                       boost::filesystem::exists(sequenceFolder_) &&
                       !boost::filesystem::is_regular_file(sequenceFolder_), false);


    associationsFilename = associationsFilename_;
    sequenceFolder = sequenceFolder_;
    vstrImageFilenamesRGB.resize(0);
    vstrImageFilenamesD.resize(0);
    vTimestamps.resize(0);

    ifstream fAssociation;
    fAssociation.open(associationsFilename.c_str());

    while (!fAssociation.eof())
    {
        string s;
        getline(fAssociation, s);

        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }

    if (vstrImageFilenamesRGB.empty())
    {
        LOG_ERROR("No images found in provided path.");
        return false;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        LOG_ERROR("Different number of images for rgb and depth.");
        return false;
    }
    else if (vstrImageFilenamesD.size() != vTimestamps.size())
    {
        LOG_ERROR("Different number of images from timestamps.");
        return false;
    }

    count = 0;
    maxCnt = vstrImageFilenamesRGB.size();
    LOG_INFO("Read done %s. Has %zu images", associationsFilename_.c_str(), maxCnt);
    return true;
}

std::shared_ptr<const FramePack> RGBD_TUM::nextFrame()
{
    auto res = std::make_shared<FramePack>();

    if (count < maxCnt)
    {

        res->frame = cv::imread(sequenceFolder + "/" + vstrImageFilenamesRGB[count],
                                cv::ImreadModes::IMREAD_UNCHANGED);
        res->depthFrame = cv::imread(sequenceFolder + "/" + vstrImageFilenamesD[count],
                                     cv::ImreadModes::IMREAD_UNCHANGED);
        res->timestamp = vTimestamps[count];

        count++;
        return res;
    }
    else if (loop)
    {
        count = 0;
        return nextFrame();
    }

    return nullptr;
}
}