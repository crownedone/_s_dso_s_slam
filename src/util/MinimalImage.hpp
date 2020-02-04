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


#pragma once

#include "util/NumType.hpp"
#include "algorithm"
#include <opencv2/core/mat.hpp>
namespace cv
{
template<> class cv::DataType<Eigen::Matrix<unsigned char, 3, 1>>
{
public:
    typedef Eigen::Matrix<unsigned char, 3, 1> value_type;
    typedef value_type work_type;
    typedef unsigned char channel_type;
    typedef value_type vec_type;
    enum
    {
        depth = CV_8U, channels = (int)sizeof(value_type) / sizeof(channel_type),
        type = CV_MAKETYPE(depth, channels), fmt = DataType<channel_type>::fmt + ((channels - 1) << 8)
    };
};

template<> class cv::DataType<Eigen::Matrix<float, 3, 1>>
{
public:
    typedef Eigen::Matrix<float, 3, 1> value_type;
    typedef value_type work_type;
    typedef float channel_type;
    typedef value_type vec_type;
    enum
    {
        depth = CV_32F, channels = (int)sizeof(value_type) / sizeof(channel_type),
        type = CV_MAKETYPE(depth, channels), fmt = DataType<channel_type>::fmt + ((channels - 1) << 8)
    };
};
}


namespace dso
{

template<typename T>
class MinimalImage
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int w;
    int h;
    cv::Mat data;

    /*
        creates minimal image with own memory
    */
    inline MinimalImage(int w_, int h_) : w(w_), h(h_)
    {
        data = cv::Mat_<T>(h, w);
        ownData = true;
    }

    /*
        creates minimal image wrapping around existing memory
    */
    inline MinimalImage(int w_, int h_, T* data_) : w(w_), h(h_)
    {
        data = cv::Mat_<T>(h, w, data_, cv::Mat::AUTO_STEP);
        ownData = false;
    }

    inline std::shared_ptr<MinimalImage> getClone()
    {
        auto clone = std::make_shared<MinimalImage>(w, h, type);
        data.copyTo(clone->data);
        return clone;
    }


    inline T& at(int x, int y)
    {
        return data.ptr<T>(y)[x];
    }

    inline T& at(int i)
    {
        return data.ptr<T>(0)[i];
    }

    inline void setBlack()
    {
        data.setTo(cv::Scalar::all(0));
    }

    inline void setPixel1(const float& u, const float& v, T val)
    {
        at(static_cast<int>(u + 0.5f), static_cast<int>(v + 0.5f)) = val;
    }


    inline void setPixel4(const float& uu, const float& vv, T val)
    {
        int u = (int)uu;
        int v = (int)vv;
        at(u + 1, v + 1) = val;
        at(u + 1, v) = val;
        at(u, v + 1) = val;
        at(u, v) = val;
    }


    inline void setPixel9(const int& u, const int& v, T val)
    {
        at(u + 1, v - 1) = val;
        at(u + 1, v) = val;
        at(u + 1, v + 1) = val;
        at(u, v - 1) = val;
        at(u, v) = val;
        at(u, v + 1) = val;
        at(u - 1, v - 1) = val;
        at(u - 1, v) = val;
        at(u - 1, v + 1) = val;
    }


    inline void setPixelCirc(const int& u, const int& v, T val)
    {
        for(int i = -3; i <= 3; i++)
        {
            at(u + 3, v + i) = val;
            at(u - 3, v + i) = val;
            at(u + 2, v + i) = val;
            at(u - 2, v + i) = val;

            at(u + i, v - 3) = val;
            at(u + i, v + 3) = val;
            at(u + i, v - 2) = val;
            at(u + i, v + 2) = val;
        }
    }


    inline void setPixelCirc(const float& uu, const float& vv, T val)
    {
        int u = static_cast<int>(uu);
        int v = static_cast<int>(vv);
        setPixelCirc(u, v, val);
    }

private:
    bool ownData;
};

typedef Eigen::Matrix<unsigned char, 3, 1> Vec3b;
typedef MinimalImage<float> MinimalImageF;
typedef MinimalImage<Vec3f> MinimalImageF3;
typedef MinimalImage<unsigned char> MinimalImageB;
typedef MinimalImage<Vec3b> MinimalImageB3;
typedef MinimalImage<unsigned short> MinimalImageB16;
}

