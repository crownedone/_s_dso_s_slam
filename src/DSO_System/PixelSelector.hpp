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
#include "util/settings.hpp"



namespace dso
{


const float minUseGrad_pixsel = 10;

enum PixelSelectorStatus { PIXSEL_VOID = 0, PIXSEL_1, PIXSEL_2, PIXSEL_3 };


struct FrameHessian;

class PixelSelector
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int makeMaps(
        std::shared_ptr<const FrameHessian> fh,
        std::vector<float>& map_out, float density, int recursionsLeft = 1, bool plot = false, float thFactor = 1);

    PixelSelector(int w, int h);
    ~PixelSelector();
    int currentPotential;

    bool allowFast;
    void makeHists(std::shared_ptr<const FrameHessian> const fh);
private:

    Eigen::Vector3i select(std::shared_ptr<const FrameHessian> fh,
                           std::vector<float>& map_out, int pot, float thFactor = 1);


    unsigned char* randomPattern;


    int* gradHist;
    float* ths;
    float* thsSmoothed;
    int thsStep;
    std::shared_ptr<const FrameHessian> gradHistFrame;
};



template<int pot>
inline int gridMaxSelection(Eigen::Vector3f* grads, std::vector<bool>& map_out, int w, int h, float THFac)
{

    map_out = std::vector<bool>(w * h, false);

    int numGood = 0;

    for(int y = 1; y < h - pot; y += pot)
    {
        for(int x = 1; x < w - pot; x += pot)
        {
            int bestXXID = -1;
            int bestYYID = -1;
            int bestXYID = -1;
            int bestYXID = -1;

            float bestXX = 0, bestYY = 0, bestXY = 0, bestYX = 0;

            Eigen::Vector3f* grads0 = grads + x + y * w;

            for(int dx = 0; dx < pot; dx++)
                for(int dy = 0; dy < pot; dy++)
                {
                    int idx = dx + dy * w;
                    Eigen::Vector3f g = grads0[idx];
                    float sqgd = g.tail<2>().squaredNorm();
                    float TH = THFac * minUseGrad_pixsel * (0.75f);

                    if(sqgd > TH * TH)
                    {
                        float agx = fabs((float)g[1]);

                        if(agx > bestXX)
                        {
                            bestXX = agx;
                            bestXXID = idx;
                        }

                        float agy = fabs((float)g[2]);

                        if(agy > bestYY)
                        {
                            bestYY = agy;
                            bestYYID = idx;
                        }

                        float gxpy = fabs((float)(g[1] - g[2]));

                        if(gxpy > bestXY)
                        {
                            bestXY = gxpy;
                            bestXYID = idx;
                        }

                        float gxmy = fabs((float)(g[1] + g[2]));

                        if(gxmy > bestYX)
                        {
                            bestYX = gxmy;
                            bestYXID = idx;
                        }
                    }
                }

            // Bool is implemented in bits in std::vector
            int idx = x + y * w;

            if(bestXXID >= 0)
            {
                if(!map_out[idx + bestXXID])
                {
                    numGood++;
                }

                map_out[idx + bestXXID] = true;

            }

            if(bestYYID >= 0)
            {
                if(!map_out[idx + bestYYID])
                {
                    numGood++;
                }

                map_out[idx + bestYYID] = true;

            }

            if(bestXYID >= 0)
            {
                if(!map_out[idx + bestXYID])
                {
                    numGood++;
                }

                map_out[idx + bestXYID] = true;

            }

            if(bestYXID >= 0)
            {
                if(!map_out[idx + bestYXID])
                {
                    numGood++;
                }

                map_out[idx + bestYXID] = true;

            }
        }
    }

    return numGood;
}


inline int gridMaxSelection(Eigen::Vector3f* grads, std::vector<bool>& map_out, int w, int h, int pot,
                            float THFac)
{

    map_out = std::vector<bool>(w * h, false);

    int numGood = 0;

    for(int y = 1; y < h - pot; y += pot)
    {
        for(int x = 1; x < w - pot; x += pot)
        {
            int bestXXID = -1;
            int bestYYID = -1;
            int bestXYID = -1;
            int bestYXID = -1;

            float bestXX = 0, bestYY = 0, bestXY = 0, bestYX = 0;

            Eigen::Vector3f* grads0 = grads + x + y * w;

            for(int dx = 0; dx < pot; dx++)
                for(int dy = 0; dy < pot; dy++)
                {
                    int idx = dx + dy * w;
                    Eigen::Vector3f g = grads0[idx];
                    float sqgd = g.tail<2>().squaredNorm();
                    float TH = THFac * minUseGrad_pixsel * (0.75f);

                    if(sqgd > TH * TH)
                    {
                        float agx = fabs((float)g[1]);

                        if(agx > bestXX)
                        {
                            bestXX = agx;
                            bestXXID = idx;
                        }

                        float agy = fabs((float)g[2]);

                        if(agy > bestYY)
                        {
                            bestYY = agy;
                            bestYYID = idx;
                        }

                        float gxpy = fabs((float)(g[1] - g[2]));

                        if(gxpy > bestXY)
                        {
                            bestXY = gxpy;
                            bestXYID = idx;
                        }

                        float gxmy = fabs((float)(g[1] + g[2]));

                        if(gxmy > bestYX)
                        {
                            bestYX = gxmy;
                            bestYXID = idx;
                        }
                    }
                }

            // Bool is implemented in bits in std::vector
            int idx = x + y * w;

            if(bestXXID >= 0)
            {
                if(!map_out[idx + bestXXID])
                {
                    numGood++;
                }

                map_out[idx + bestXXID] = true;

            }

            if(bestYYID >= 0)
            {
                if(!map_out[idx + bestYYID])
                {
                    numGood++;
                }

                map_out[idx + bestYYID] = true;

            }

            if(bestXYID >= 0)
            {
                if(!map_out[idx + bestXYID])
                {
                    numGood++;
                }

                map_out[idx + bestXYID] = true;

            }

            if(bestYXID >= 0)
            {
                if(!map_out[idx + bestYXID])
                {
                    numGood++;
                }

                map_out[idx + bestYXID] = true;

            }
        }
    }

    return numGood;
}


inline int makePixelStatus(Eigen::Vector3f* grads, std::vector<bool>& map, int w, int h, float desiredDensity,
                           int recsLeft = 5, float THFac = 1)
{
    if(sparsityFactor < 1)
    {
        sparsityFactor = 1;
    }

    int numGoodPoints;


    if(sparsityFactor == 1)
    {
        numGoodPoints = gridMaxSelection<1>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 2)
    {
        numGoodPoints = gridMaxSelection<2>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 3)
    {
        numGoodPoints = gridMaxSelection<3>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 4)
    {
        numGoodPoints = gridMaxSelection<4>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 5)
    {
        numGoodPoints = gridMaxSelection<5>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 6)
    {
        numGoodPoints = gridMaxSelection<6>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 7)
    {
        numGoodPoints = gridMaxSelection<7>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 8)
    {
        numGoodPoints = gridMaxSelection<8>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 9)
    {
        numGoodPoints = gridMaxSelection<9>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 10)
    {
        numGoodPoints = gridMaxSelection<10>(grads, map, w, h, THFac);
    }
    else if(sparsityFactor == 11)
    {
        numGoodPoints = gridMaxSelection<11>(grads, map, w, h, THFac);
    }
    else
    {
        numGoodPoints = gridMaxSelection(grads, map, w, h, sparsityFactor, THFac);
    }


    /*
        #points is approximately proportional to sparsityFactor^2.
    */

    float quotia = numGoodPoints / (float)(desiredDensity);

    int newSparsity = static_cast<int>((sparsityFactor * sqrtf(quotia)) + 0.7f);


    if(newSparsity < 1)
    {
        newSparsity = 1;
    }


    float oldTHFac = THFac;

    if(newSparsity == 1 && sparsityFactor == 1)
    {
        THFac = 0.5;
    }


    if((abs(newSparsity - sparsityFactor) < 1 && THFac == oldTHFac) ||
       ( quotia > 0.8 &&  1.0f / quotia > 0.8) ||
       recsLeft == 0)
    {
        LOG_INFO(" \n");
        //all good
        sparsityFactor = newSparsity;
        return numGoodPoints;
    }
    else
    {
        LOG_INFO(" -> re-evaluate!");
        // re-evaluate.
        sparsityFactor = newSparsity;
        return makePixelStatus(grads, map, w, h, desiredDensity, recsLeft - 1, THFac);
    }
}

}

