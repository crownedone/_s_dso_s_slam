///**
//    This file is part of DSO.
//
//    Copyright 2016 Technical University of Munich and Intel.
//    Developed by Jakob Engel <engelj at in dot tum dot de>,
//    for more information see <http://vision.in.tum.de/dso>.
//    If you use this code, please cite the respective publications as
//    listed on the above website.
//
//    DSO is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    DSO is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with DSO. If not, see <http://www.gnu.org/licenses/>.
//*/
//
//
//#pragma once
//#include "util/settings.hpp"
//#include "util/globalFuncs.hpp"
//#include "util/globalCalib.hpp"
//
//#include <sstream>
//#include <fstream>
//#include <algorithm>
//
//#include "util/Undistort.hpp"
//#include <opencv2/imgcodecs.hpp>
//#if HAS_ZIPLIB
//    #include "zip.hpp"
//#endif
//
//#include <boost/thread.hpp>
//#include <boost/filesystem.hpp>
//#include <boost/range/iterator_range.hpp>
//
//using namespace dso;
//
//class ImageFolderReader
//{
//public:
//
//    bool valid = false;
//
//    ImageFolderReader(const std::string& path0, bool readStereo = false)
//    {
//        path = path0;
//
//        if (!boost::filesystem::exists(path) | !boost::filesystem::is_directory(path))
//        {
//            LOG_ERROR("Not an existing directory: %s", path.string().c_str());
//            exit(EXIT_FAILURE);
//        }
//
//        images = path / "images";
//        calibfile = path / "camera.txt";
//        gamma = path / "pcalib.txt";
//        vignette = path / "vignette.png";
//
//
//#if HAS_ZIPLIB
//        ziparchive = 0;
//        databuffer = 0;
//#endif
//
//        isZipped = (!boost::filesystem::exists(images) && boost::filesystem::exists(path / "images.zip"));
//
//        if(isZipped)
//        {
//            images = path / "images.zip";
//#if HAS_ZIPLIB
//            int ziperror = 0;
//            ziparchive = zip_open(images.string().c_str(),  ZIP_RDONLY, &ziperror);
//
//            if(ziperror != 0)
//            {
//                LOG_ERROR("%d reading archive %s!\n", ziperror, path.c_str());
//                exit(EXIT_FAILURE);
//            }
//
//            files.clear();
//            int numEntries = zip_get_num_entries(ziparchive, 0);
//
//            for(int k = 0; k < numEntries; k++)
//            {
//                const char* name = zip_get_name(ziparchive, k,  ZIP_FL_ENC_STRICT);
//                std::string nstr = std::string(name);
//
//                if(nstr == "." || nstr == "..")
//                {
//                    continue;
//                }
//
//                files.push_back(name);
//            }
//
//            LOG_INFO("got %d entries and %d files!\n", numEntries, (int)files.size());
//            std::sort(files.begin(), files.end());
//#else
//            LOG_ERROR("cannot read .zip archive, as compile without ziplib!\n");
//            exit(EXIT_FAILURE);
//#endif
//        }
//        else
//        {
//            CHECK_ARG_FATAL(boost::filesystem::exists(images) && boost::filesystem::is_directory(images));
//
//            for (auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(images), {}))
//                files.push_back(entry.path().filename().string());
//
//            std::sort(files.begin(), files.end());
//
//            for (int i = 0; i < files.size(); ++i)
//            {
//                files[i] = images.string() + "/" + files[i];
//                LOG_INFO_1("%s \n", files[i].c_str());
//            }
//        }
//
//
//        undistort = Undistort::getUndistorterForFile(calibfile.string(), gamma.string(), vignette.string());
//
//
//        widthOrg = undistort->getOriginalSize()[0];
//        heightOrg = undistort->getOriginalSize()[1];
//        width = undistort->getSize()[0];
//        height = undistort->getSize()[1];
//
//        // load timestamps if possible.
//        loadTimestamps();
//        LOG_INFO("ImageFolderReader: got %d files in %s!\n", (int)files.size(), images.string().c_str());
//
//    }
//    ~ImageFolderReader()
//    {
//#if HAS_ZIPLIB
//
//        if(ziparchive != 0)
//        {
//            zip_close(ziparchive);
//        }
//
//        if(databuffer != 0)
//        {
//            delete databuffer;
//        }
//
//#endif
//
//
//        delete undistort;
//    };
//
//    Eigen::VectorXf getOriginalCalib()
//    {
//        return undistort->getOriginalParameter().cast<float>();
//    }
//    Eigen::Vector2i getOriginalDimensions()
//    {
//        return  undistort->getOriginalSize();
//    }
//
//    void getCalibMono(Eigen::Matrix3f& K, int& w, int& h)
//    {
//        K = undistort->getK().cast<float>();
//        w = undistort->getSize()[0];
//        h = undistort->getSize()[1];
//    }
//
//    void setGlobalCalibration()
//    {
//        int w_out, h_out;
//        Eigen::Matrix3f K;
//        getCalibMono(K, w_out, h_out);
//        setGlobalCalib(w_out, h_out, K);
//    }
//
//    int getNumImages()
//    {
//        return static_cast<int>(files.size());
//    }
//
//    double getTimestamp(int id)
//    {
//        if(timestamps.size() == 0)
//        {
//            return id * 0.1f;
//        }
//
//        if(id >= (int)timestamps.size())
//        {
//            return 0;
//        }
//
//        if(id < 0)
//        {
//            return 0;
//        }
//
//        return timestamps[id];
//    }
//
//
//    void prepImage(int id, bool as8U = false)
//    {
//
//    }
//
//
//    cv::Mat getImageRaw(int id)
//    {
//        return getImageRaw_internal(id, 0);
//    }
//
//    ImageAndExposure* getImage(int id, bool forceLoadDirectly = false)
//    {
//        return getImage_internal(id, 0);
//    }
//
//
//    inline float* getPhotometricGamma()
//    {
//        if(undistort == 0 || undistort->photometricUndist == 0)
//        {
//            return 0;
//        }
//
//        return undistort->photometricUndist->getG();
//    }
//
//
//    // undistorter. [0] always exists, [1-2] only when MT is enabled.
//    Undistort* undistort;
//private:
//
//
//    cv::Mat getImageRaw_internal(int id, int unused)
//    {
//        if(!isZipped)
//        {
//            // CHANGE FOR ZIP FILE
//            return cv::imread(files[id], cv::IMREAD_GRAYSCALE);
//        }
//        else
//        {
//#if HAS_ZIPLIB
//
//            if(databuffer == 0)
//            {
//                databuffer = new char[widthOrg * heightOrg * 6 + 10000];
//            }
//
//            zip_file_t* fle = zip_fopen(ziparchive, files[id].c_str(), 0);
//            long readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 6 + 10000);
//
//            if(readbytes > (long)widthOrg * heightOrg * 6)
//            {
//                LOG_INFO("read %ld/%ld bytes for file %s. increase buffer!!\n", readbytes,
//                         (long)widthOrg * heightOrg * 6 + 10000, files[id].c_str());
//                delete[] databuffer;
//                databuffer = new char[(long)widthOrg * heightOrg * 30];
//                fle = zip_fopen(ziparchive, files[id].c_str(), 0);
//                readbytes = zip_fread(fle, databuffer, (long)widthOrg * heightOrg * 30 + 10000);
//
//                if(readbytes > (long)widthOrg * heightOrg * 30)
//                {
//                    LOG_INFO("buffer still to small (read %ld/%ld). abort.\n", readbytes,
//                             (long)widthOrg * heightOrg * 30 + 10000);
//                    exit(1);
//                }
//            }
//
//            return IOWrap::readStreamBW_8U(databuffer, readbytes);
//#else
//            LOG_ERROR("cannot read .zip archive, as compiled without ziplib!");
//            exit(1);
//#endif
//        }
//    }
//
//
//    std::shared_ptr<ImageAndExposure> getImage_internal(int id, int unused)
//    {
//
//        cv::Mat minimg = getImageRaw_internal(id, 0);
//        auto ret2 = undistort->undistort(
//                        minimg,
//                        (exposures.size() == 0 ? 1.0f : exposures[id]),
//                        (timestamps.size() == 0 ? 0.0 : timestamps[id]));
//        return ret2;
//    }
//
//    inline void loadTimestamps()
//    {
//        std::ifstream tr;
//        boost::filesystem::path timesFile = path / "/times.txt";
//        tr.open(timesFile.string().c_str());
//
//        while(!tr.eof() && tr.good())
//        {
//            std::string line;
//            char buf[1000];
//            tr.getline(buf, 1000);
//
//            int id;
//            double stamp;
//            float exposure = 0;
//
//            if(3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
//            {
//                timestamps.push_back(stamp);
//                exposures.push_back(exposure);
//            }
//
//            else if(2 == sscanf(buf, "%d %lf", &id, &stamp))
//            {
//                timestamps.push_back(stamp);
//                exposures.push_back(exposure);
//            }
//        }
//
//        tr.close();
//
//        // check if exposures are correct, (possibly skip)
//        bool exposuresGood = ((int)exposures.size() == (int)getNumImages()) ;
//
//        for(int i = 0; i < (int)exposures.size(); i++)
//        {
//            if(exposures[i] == 0)
//            {
//                // fix!
//                float sum = 0, num = 0;
//
//                if(i > 0 && exposures[i - 1] > 0)
//                {
//                    sum += exposures[i - 1];
//                    num++;
//                }
//
//                if(i + 1 < (int)exposures.size() && exposures[i + 1] > 0)
//                {
//                    sum += exposures[i + 1];
//                    num++;
//                }
//
//                if(num > 0)
//                {
//                    exposures[i] = sum / num;
//                }
//            }
//
//            if(exposures[i] == 0)
//            {
//                exposuresGood = false;
//            }
//        }
//
//
//        if((int)getNumImages() != (int)timestamps.size())
//        {
//            LOG_INFO("set timestamps and exposures to zero!\n");
//            exposures.clear();
//            timestamps.clear();
//        }
//
//        if((int)getNumImages() != (int)exposures.size() || !exposuresGood)
//        {
//            LOG_INFO("set EXPOSURES to zero!\n");
//            exposures.clear();
//        }
//
//        LOG_INFO("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(),
//                 (int)timestamps.size(), (int)exposures.size());
//    }
//
//
//
//
//    std::vector<ImageAndExposure*> preloadedImages;
//    std::vector<std::string> files;
//    std::vector<double> timestamps;
//    std::vector<float> exposures;
//
//    int width, height;
//    int widthOrg, heightOrg;
//
//    boost::filesystem::path path;
//    boost::filesystem::path images;
//    boost::filesystem::path calibfile;
//    boost::filesystem::path gamma;
//    boost::filesystem::path vignette;
//
//    bool isZipped;
//
//#if HAS_ZIPLIB
//    zip_t* ziparchive;
//    char* databuffer;
//#endif
//};
//
