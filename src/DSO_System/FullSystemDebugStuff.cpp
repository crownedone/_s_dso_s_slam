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


/*
    KFBuffer.cpp

    Created on: Jan 7, 2014
        Author: engelj
*/

#include "DSO_system/FullSystem.hpp"

#include "util/globalFuncs.hpp"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/Output3D.hpp"
#include "util/globalCalib.hpp"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <algorithm>

#include "DSO_system/ImmaturePoint.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace dso
{


void FullSystem::debugPlotTracking()
{
    if(disableAllDisplay)
    {
        return;
    }

    if(!setting_render_plotTrackingFull)
    {
        return;
    }

    int wh = hG[0] * wG[0];

    int idx = 0;

    for(auto& f : frameHessians)
    {
        std::vector<cv::Mat> images;

        // make images for all frames. will be deleted by the FrameHessian's destructor.
        for(auto& f2 : frameHessians)
        {
            if(f2->debugImage.empty())
            {
                f2->debugImage = cv::Mat(hG[0], wG[0], CV_8UC3);
            }

            cv::Mat debugImage = f2->debugImage;
            images.push_back(debugImage);

            const Eigen::Vector3f* fd = f2->dI_ptr;

            Vec2 affL = AffLight::fromToVecExposure(f2->ab_exposure, f->ab_exposure, f2->aff_g2l(),
                                                    f->aff_g2l());

            for(int i = 0; i < wh; i++)
            {
                // BRIGHTNESS TRANSFER
                float colL = affL[0] * fd[i][0] + affL[1];

                if(colL < 0)
                {
                    colL = 0;
                }

                if(colL > 255)
                {
                    colL = 255;
                }

                debugImage.at<cv::Vec3b>(i) = cv::Vec3b(colL, colL, colL);
            }
        }


        for(PointHessian* ph : f->pointHessians)
        {
            assert(ph->status == PointHessian::ACTIVE);

            if(ph->status == PointHessian::ACTIVE || ph->status == PointHessian::MARGINALIZED)
            {
                for(PointFrameResidual* r : ph->residuals)
                {
                    r->debugPlot();
                }

                cv::circle(f->debugImage, cv::Point(ph->u + 0.5, ph->v + 0.5), 3, makeRainbow3B(ph->idepth_scaled),
                           cv::FILLED);
            }
        }

        char buf[100];
        snprintf(buf, 100, "IMG %d", idx);
        Viewer::displayImageStitch(buf, images);
        idx++;
    }

    Viewer::waitKey(0);

}


void FullSystem::debugPlot(std::string name)
{
    if(disableAllDisplay)
    {
        return;
    }

    if(!setting_render_renderWindowFrames)
    {
        return;
    }

    std::vector<cv::Mat> images;

    float minID = 0, maxID = 0;

    if((int)(freeDebugParam5 + 0.5f) == 7 || (debugSaveImages && false))
    {
        std::vector<float> allID;

        for(unsigned int f = 0; f < frameHessians.size(); f++)
        {
            for(PointHessian* ph : frameHessians[f]->pointHessians)
                if(ph != 0)
                {
                    allID.push_back(ph->idepth_scaled);
                }

            for(PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
                if(ph != 0)
                {
                    allID.push_back(ph->idepth_scaled);
                }

            for(PointHessian* ph : frameHessians[f]->pointHessiansOut)
                if(ph != 0)
                {
                    allID.push_back(ph->idepth_scaled);
                }
        }

        std::sort(allID.begin(), allID.end());
        int n = allID.size() - 1;
        minID = allID[(int)(n * 0.05)];
        maxID = allID[(int)(n * 0.95)];


        // slowly adapt: change by maximum 10% of old span.
        float maxChange = 0.1 * (maxIdJetVisDebug - minIdJetVisDebug);

        if(maxIdJetVisDebug < 0  || minIdJetVisDebug < 0 )
        {
            maxChange = 1e5;
        }


        if(minID < minIdJetVisDebug - maxChange)
        {
            minID = minIdJetVisDebug - maxChange;
        }

        if(minID > minIdJetVisDebug + maxChange)
        {
            minID = minIdJetVisDebug + maxChange;
        }


        if(maxID < maxIdJetVisDebug - maxChange)
        {
            maxID = maxIdJetVisDebug - maxChange;
        }

        if(maxID > maxIdJetVisDebug + maxChange)
        {
            maxID = maxIdJetVisDebug + maxChange;
        }

        maxIdJetVisDebug = maxID;
        minIdJetVisDebug = minID;

    }












    int wh = hG[0] * wG[0];

    for(unsigned int f = 0; f < frameHessians.size(); f++)
    {
        cv::Mat img(hG[0], wG[0], CV_8UC3);
        images.push_back(img);
        //float* fd = frameHessians[f]->I;
        const Eigen::Vector3f* fd = frameHessians[f]->dI_ptr;


        for(int i = 0; i < wh; i++)
        {
            int c = fd[i][0] * 0.9f;

            if(c > 255)
            {
                c = 255;
            }

            img.at<cv::Vec3b>(i) = cv::Vec3b(c, c, c);
        }

        if((int)(freeDebugParam5 + 0.5f) == 0)
        {
            for(PointHessian* ph : frameHessians[f]->pointHessians)
            {
                if(ph == 0)
                {
                    continue;
                }

                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, makeRainbow3B(ph->idepth_scaled),
                           cv::FILLED);
            }

            for(PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
            {
                if(ph == 0)
                {
                    continue;
                }

                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, makeRainbow3B(ph->idepth_scaled),
                           cv::FILLED);
            }

            for(PointHessian* ph : frameHessians[f]->pointHessiansOut)
            {
                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, cv::Vec3b(255, 255, 255),
                           cv::FILLED);
            }
        }
        else if((int)(freeDebugParam5 + 0.5f) == 1)
        {
            for(PointHessian* ph : frameHessians[f]->pointHessians)
            {
                if(ph == 0)
                {
                    continue;
                }

                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, makeRainbow3B(ph->idepth_scaled),
                           cv::FILLED);
            }

            for(PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
            {
                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, cv::Vec3b(0, 0, 0),
                           cv::FILLED);
            }

            for(PointHessian* ph : frameHessians[f]->pointHessiansOut)
            {
                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, cv::Vec3b(255, 255, 255),
                           cv::FILLED);
            }
        }
        else if((int)(freeDebugParam5 + 0.5f) == 2)
        {

        }
        else if((int)(freeDebugParam5 + 0.5f) == 3)
        {
            for(ImmaturePoint* ph : frameHessians[f]->immaturePoints)
            {
                if(ph == 0)
                {
                    continue;
                }

                if(ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD ||
                   ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED ||
                   ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION)
                {
                    if(!std::isfinite(ph->idepth_max))
                    {
                        cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, cv::Vec3b(0, 0, 0),
                                   cv::FILLED);
                    }
                    else
                    {
                        cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3,
                                   makeRainbow3B((ph->idepth_min + ph->idepth_max) * 0.5f),
                                   cv::FILLED);
                    }
                }
            }
        }
        else if((int)(freeDebugParam5 + 0.5f) == 4)
        {
            for(ImmaturePoint* ph : frameHessians[f]->immaturePoints)
            {
                if(ph == 0)
                {
                    continue;
                }

                cv::Vec3b col;

                switch (ph->lastTraceStatus)
                {
                case ImmaturePointStatus::IPS_GOOD:
                    col = cv::Vec3b(0, 255, 0);
                    break;

                case ImmaturePointStatus::IPS_OOB:
                    col = cv::Vec3b(255, 0, 0);
                    break;

                case ImmaturePointStatus::IPS_OUTLIER:
                    col = cv::Vec3b(0, 0, 255);
                    break;

                case ImmaturePointStatus::IPS_SKIPPED:
                    col = cv::Vec3b(255, 255, 0);
                    break;

                case ImmaturePointStatus::IPS_BADCONDITION:
                    col = cv::Vec3b(255, 255, 255);
                    break;

                case ImmaturePointStatus::IPS_UNINITIALIZED:
                    col = cv::Vec3b(0, 0, 0);
                    break;
                }

                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, col, cv::FILLED);
            }
        }
        else if((int)(freeDebugParam5 + 0.5f) == 5)
        {
            for(ImmaturePoint* ph : frameHessians[f]->immaturePoints)
            {
                if(ph == 0)
                {
                    continue;
                }

                if(ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED)
                {
                    continue;
                }

                float d = freeDebugParam1 * (sqrtf(ph->quality) - 1);

                if(d < 0)
                {
                    d = 0;
                }

                if(d > 1)
                {
                    d = 1;
                }

                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, cv::Vec3b(0, d * 255, (1 - d) * 255),
                           cv::FILLED);
            }

        }
        else if((int)(freeDebugParam5 + 0.5f) == 6)
        {
            for(PointHessian* ph : frameHessians[f]->pointHessians)
            {
                if(ph == 0)
                {
                    continue;
                }

                cv::Vec3b col(0, 0, 0);

                if(ph->my_type == 0)
                {
                    col = cv::Vec3b(255, 0, 255);
                }

                if(ph->my_type == 1)
                {
                    col = cv::Vec3b(255, 0, 0);
                }

                if(ph->my_type == 2)
                {
                    col = cv::Vec3b(0, 0, 255);
                }

                if(ph->my_type == 3)
                {
                    col = cv::Vec3b(0, 255, 255);
                }

                if(col != cv::Vec3b(0, 0, 0))
                {
                    cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, col, cv::FILLED);
                }
            }

            for(PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
            {
                if(ph == 0)
                {
                    continue;
                }

                cv::Vec3b col(0, 0, 0);

                if(ph->my_type == 0)
                {
                    col = cv::Vec3b(255, 0, 255);
                }

                if(ph->my_type == 1)
                {
                    col = cv::Vec3b(255, 0, 0);
                }

                if(ph->my_type == 2)
                {
                    col = cv::Vec3b(0, 0, 255);
                }

                if(ph->my_type == 3)
                {
                    col = cv::Vec3b(0, 255, 255);
                }

                if (col != cv::Vec3b(0, 0, 0))
                {
                    cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3, col, cv::FILLED);
                }
            }

        }

        if((int)(freeDebugParam5 + 0.5f) == 7)
        {
            for(PointHessian* ph : frameHessians[f]->pointHessians)
            {
                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3,
                           makeJet3B((ph->idepth_scaled - minID) / ((maxID - minID))), cv::FILLED);
            }

            for(PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
            {
                if(ph == 0)
                {
                    continue;
                }

                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3,
                           cv::Vec3b(0, 0, 0), cv::FILLED);
            }
        }
    }

    Viewer::displayImageStitch(name.c_str(), images);
    Viewer::waitKey(5);

    if((debugSaveImages && false))
    {
        for(unsigned int f = 0; f < frameHessians.size(); f++)
        {
            cv::Mat img(hG[0], wG[0], CV_8UC3);
            const Eigen::Vector3f* fd = frameHessians[f]->dI_ptr;

            for(int i = 0; i < wh; i++)
            {
                int c = fd[i][0] * 0.9f;

                if(c > 255)
                {
                    c = 255;
                }

                img.at<cv::Vec3b>(i) = cv::Vec3b(c, c, c);
            }

            for(PointHessian* ph : frameHessians[f]->pointHessians)
            {
                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3,
                           makeJet3B((ph->idepth_scaled - minID) / ((maxID - minID))), cv::FILLED);
            }

            for(PointHessian* ph : frameHessians[f]->pointHessiansMarginalized)
            {
                if(ph == 0)
                {
                    continue;
                }

                cv::circle(img, cv::Point(ph->u + 0.5f, ph->v + 0.5f), 3,
                           cv::Vec3b(0, 0, 0), cv::FILLED);
            }

            char buf[1000];
            snprintf(buf, 1000, "images_out/kf_%05d_%05d_%02d.png",
                     frameHessians.back()->shell->id,  frameHessians.back()->frameID, f);
            cv::imwrite(buf, img);
        }
    }




}






}
