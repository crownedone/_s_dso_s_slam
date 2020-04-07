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



#include "IOWrapper/ImageDisplay.hpp"

#include <opencv2/highgui/highgui.hpp>

#include <string>
#include <unordered_set>

#include <boost/thread.hpp>

#include "util/settings.hpp"

namespace dso
{


namespace IOWrap
{

std::unordered_set<std::string> openWindows;
boost::mutex openCVdisplayMutex;



void displayImage(const char* windowName, const cv::Mat& img, bool autoSize)
{
    if(disableAllDisplay)
    {
        return;
    }

    cv::Mat image = img;

    if (image.depth() == CV_32F)
    {
        image = img * (1 / 254.0f);
    }

    boost::unique_lock<boost::mutex> lock(openCVdisplayMutex);

    if(!autoSize)
    {
        if(openWindows.find(windowName) == openWindows.end())
        {
            cv::namedWindow(windowName, cv::WINDOW_NORMAL);
            cv::resizeWindow(windowName, image.cols, image.rows);
            openWindows.insert(windowName);
        }
    }

    cv::imshow(windowName, image);
}


void displayImageStitch(const char* windowName, const std::vector<cv::Mat> images, int cc, int rc)
{
    if(disableAllDisplay)
    {
        return;
    }

    if(images.size() == 0)
    {
        return;
    }

    // get dimensions.
    int w = images[0].cols;
    int h = images[0].rows;

    int num = std::max((int)setting_maxFrames, (int)images.size());

    // get optimal dimensions.
    int bestCC = 0;
    float bestLoss = 1e10;

    for(int cc = 1; cc < 10; cc++)
    {
        int ww = w * cc;
        int hh = h * ((num + cc - 1) / cc);


        float wLoss = ww / 16.0f;
        float hLoss = hh / 10.0f;
        float loss = std::max(wLoss, hLoss);

        if(loss < bestLoss)
        {
            bestLoss = loss;
            bestCC = cc;
        }
    }

    int bestRC = ((num + bestCC - 1) / bestCC);

    if(cc != 0)
    {
        bestCC = cc;
        bestRC = rc;
    }

    cv::Mat stitch = cv::Mat(bestRC * h, bestCC * w, images[0].type());
    stitch.setTo(0);

    for(int i = 0; i < (int)images.size() && i < bestCC * bestRC; i++)
    {
        int c = i % bestCC;
        int r = i / bestCC;

        cv::Mat roi = stitch(cv::Rect(c * w, r * h, w, h));
        images[i].copyTo(roi);
    }

    displayImage(windowName, stitch, false);
}

int waitKey(int milliseconds)
{
    if(disableAllDisplay)
    {
        return 0;
    }

    boost::unique_lock<boost::mutex> lock(openCVdisplayMutex);
    return cv::waitKey(milliseconds);
}

void closeAllWindows()
{
    if(disableAllDisplay)
    {
        return;
    }

    boost::unique_lock<boost::mutex> lock(openCVdisplayMutex);
    cv::destroyAllWindows();
    openWindows.clear();
}
}

}
