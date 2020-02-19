#pragma once
#include "Input.hpp"
#include <opencv2/imgcodecs.hpp>

#define INPUT_CHECK_SIZE(I) if(I < static_cast<unsigned int>(vstrImageFilenames.size())) return false;

int Input::getNumImages() const
{
    return static_cast<int>(vstrImageFilenames.size());
}

bool Input::getImageAt(unsigned int it, cv::Mat& img, double& timestamp)
{
    INPUT_CHECK_SIZE(it)

    img = cv::imread(sequenceFolder + "/" + vstrImageFilenames[it], cv::ImreadModes::IMREAD_UNCHANGED);

    return false;
}

bool Input::getImage1At(unsigned int it, cv::Mat& img, double& timestamp)
{
    INPUT_CHECK_SIZE(it)

    img = cv::imread(sequenceFolder1 + "/" + vstrImageFilenames1[it],
                     cv::ImreadModes::IMREAD_UNCHANGED);

    return false;
}

double Input::getTimestampAt(unsigned int i)
{
    INPUT_CHECK_SIZE(i)

    return vTimestamps[i];
}
