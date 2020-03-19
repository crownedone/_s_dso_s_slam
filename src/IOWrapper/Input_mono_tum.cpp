#include "Input.hpp"
#include "Logging.hpp"

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

#include <opencv2/imgcodecs.hpp>

namespace IO
{

bool Mono_TUM::open(const std::string& path0, bool looping /*= false*/)
{
    path = path0;

    CHECK_ARG_WITH_RET(boost::filesystem::exists(path) && boost::filesystem::is_directory(path), false);

    images = path + "/images";

    images1 = path + "/images1";
    bool hasStereo = boost::filesystem::exists(images1) && !boost::filesystem::is_empty(images1);

    // no zip treatment
    //bool isZipped = (!boost::filesystem::exists(images) && boost::filesystem::exists(path / "images.zip"));

    CHECK_ARG_WITH_RET(boost::filesystem::exists(images) &&
                       boost::filesystem::is_directory(images), false);

    for (auto& entry :
         boost::make_iterator_range(boost::filesystem::directory_iterator(boost::filesystem::path(images)), {}))
    {
        files.push_back(entry.path().filename().string());
    }

    std::sort(files.begin(), files.end());

    for (int i = 0; i < files.size(); ++i)
    {
        if (hasStereo)
        {
            files1.push_back(images1 + "/" + files[i]);
        }

        files[i] = images + "/" + files[i];
        LOG_INFO_1("%s \n", files[i].c_str());
    }

    // load timestamps if possible.
    if (!loadTimestamps())
    {
        LOG_ERROR("Cannot read timestamps file");
        return false;
    }

    LOG_INFO("Input Mono_TUM: got %d files in %s!\n", (int)files.size(), images.c_str());

    return true;
}

bool Mono_TUM::loadTimestamps()
{
    boost::filesystem::path timesFile = path + "/times.txt";
    boost::filesystem::path timesFile1 = path + "/times1.txt";


    auto loadTS = [](boost::filesystem::path timesFile, std::vector<std::string>& files, std::vector<float>& exposures,
                     std::vector<double>& timestamps, size_t& maxCnt)
    {
        CHECK_ARG_WITH_RET(boost::filesystem::exists(timesFile) &&
                           boost::filesystem::is_regular_file(timesFile), false);
        std::ifstream tr;

        tr.open(timesFile.string().c_str());

        while (!tr.eof() && tr.good())
        {
            std::string line;
            char buf[1000];
            tr.getline(buf, 1000);

            int id;
            double stamp;
            float exposure = 0;

            if (3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
            {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }
            else if (2 == sscanf(buf, "%d %lf", &id, &stamp))
            {
                timestamps.push_back(stamp);
                exposures.push_back(exposure);
            }
        }

        tr.close();

        // check if exposures are correct, (possibly skip)
        bool exposuresGood = ((int)exposures.size() == static_cast<int>(files.size()));

        if (!exposuresGood)
        {
            LOG_WARNING("Clearing exposures!");
            exposures.clear();
        }

        if (files.size() != timestamps.size())
        {
            LOG_WARNING("Clearing timestamps!");
            timestamps.clear();
        }

        // fix exposures (really?)
        for (int i = 0; i < (int)exposures.size(); i++)
        {
            if (exposures[i] == 0)
            {
                // fix!
                float sum = 0, num = 0;

                if (i > 0 && exposures[i - 1] > 0)
                {
                    sum += exposures[i - 1];
                    num++;
                }

                if (i + 1 < (int)exposures.size() && exposures[i + 1] > 0)
                {
                    sum += exposures[i + 1];
                    num++;
                }

                if (num > 0)
                {
                    exposures[i] = sum / num;
                }
            }

            if (exposures[i] == 0)
            {
                exposuresGood = false;
            }
        }


        LOG_INFO("got %d images and %d timestamps and %d exposures.!\n", static_cast<int>(files.size()),
                 (int)timestamps.size(), (int)exposures.size());
        maxCnt = files.size();
        return true;
    };
    bool success = loadTS(timesFile, files, exposures, timestamps, maxCnt);

    if (boost::filesystem::exists(timesFile1))
    {
        size_t otherMaxCnt;
        success &= loadTS(timesFile1, files1, exposures1, timestamps1, otherMaxCnt);
        LOG_ERROR_IF(maxCnt != otherMaxCnt, "Invalid stuff with stereo read");
    }

    return success;
}

std::shared_ptr<const FramePack> Mono_TUM::nextFrame()
{
    auto res = std::make_shared<FramePack>();

    if (count < maxCnt)
    {
        res->frame = cv::imread(files[count], cv::IMREAD_GRAYSCALE);
        res->timestamp = (timestamps.size() == 0 ? 0.0 : timestamps[count]);
        res->exposure = (exposures.size() == 0 ? 1.0f : exposures[count]);

        // Stereo
        if (!files1.empty())
        {
            res->frame_slave1 = cv::imread(files1[count], cv::IMREAD_GRAYSCALE);
            res->timestamp_slave1 = (timestamps1.size() == 0 ? 0.0 : timestamps1[count]);
            res->exposure_slave1 = (exposures1.size() == 0 ? 1.0f : exposures1[count]);
        }

        res->id = count;
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