#pragma once

#include <string>
#include <opencv2/core/core.hpp>

class Input
{
public:
    virtual bool read(const std::string& associationsFilename,
                      const std::string& sequenceFolder = "") {};
    virtual int getNumImages() const;

    bool getImageAt(unsigned int it, cv::Mat& img, double& timestamp);
    bool getImage1At(unsigned int it, cv::Mat& img, double& timestamp);

    double getTimestampAt(unsigned int i);
protected:
    std::string associationsFilename;
    std::string sequenceFolder;
    std::vector<std::string> vstrImageFilenames;
    std::vector<std::string> vstrImageFilenames1;
    std::vector<double> vTimestamps;


    std::thread m_CaptureThread;
};


class RGBD_TUM : public Input
{
public:
    bool read(const std::string& associationsFilename,
              const std::string& sequenceFolder = "") override;

};