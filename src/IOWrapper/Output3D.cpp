#include "Output3D.hpp"
#include <unordered_set>
#include <opencv2/highgui.hpp>

namespace Viewer
{
std::unordered_set<std::string> openWindows;
std::mutex openCVdisplayMutex;


void displayImage(const char* windowName, const cv::Mat& img, bool autoSize)
{
    cv::Mat image = img;

    if (image.depth() == CV_32F)
    {
        image = img * (1 / 254.0f);
    }

    std::unique_lock<std::mutex> lock(openCVdisplayMutex);

    if (!autoSize)
    {
        if (openWindows.find(windowName) == openWindows.end())
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
    if (images.size() == 0)
    {
        return;
    }

    // get dimensions.
    int w = images[0].cols;
    int h = images[0].rows;

    int num = std::max(20, (int)images.size());

    // get optimal dimensions.
    int bestCC = 0;
    float bestLoss = 1e10;

    for (int cc = 1; cc < 10; cc++)
    {
        int ww = w * cc;
        int hh = h * ((num + cc - 1) / cc);


        float wLoss = ww / 16.0f;
        float hLoss = hh / 10.0f;
        float loss = std::max(wLoss, hLoss);

        if (loss < bestLoss)
        {
            bestLoss = loss;
            bestCC = cc;
        }
    }

    int bestRC = ((num + bestCC - 1) / bestCC);

    if (cc != 0)
    {
        bestCC = cc;
        bestRC = rc;
    }

    cv::Mat stitch = cv::Mat(bestRC * h, bestCC * w, images[0].type());
    stitch.setTo(0);

    for (int i = 0; i < (int)images.size() && i < bestCC * bestRC; i++)
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
    std::unique_lock<std::mutex> lock(openCVdisplayMutex);
    return cv::waitKey(milliseconds);
}

void closeAllWindows()
{
    std::unique_lock<std::mutex> lock(openCVdisplayMutex);
    cv::destroyAllWindows();
    openWindows.clear();
}

bool Output3D::needPushDepthImage()
{
    return false;
}

}