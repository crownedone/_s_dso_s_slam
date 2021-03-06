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
#include "util/globalCalib.hpp"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "DSO_system/ResidualProjections.hpp"
#include "DSO_system/ImmaturePoint.hpp"

#include "OptimizationBackend/EnergyFunctional.hpp"
#include "OptimizationBackend/EnergyFunctionalStructs.hpp"

#include "IOWrapper/Output3D.hpp"

#include "DSO_system/CoarseTracker.hpp"

namespace dso
{



void FullSystem::flagFramesForMarginalization(std::shared_ptr<FrameHessian> newFH)
{
    if(setting_minFrameAge > setting_maxFrames)
    {
        for(int i = setting_maxFrames; i < (int)frameHessians.size(); i++)
        {
            std::shared_ptr<FrameHessian> fh = frameHessians[i - setting_maxFrames];
            fh->flaggedForMarginalization = true;
        }

        return;
    }


    int flagged = 0;

    // marginalize all frames that have not enough points.
    for(int i = 0; i < (int)frameHessians.size(); i++)
    {
        auto fh = frameHessians[i];
        int in = fh->pointHessians.size() + fh->immaturePoints.size();
        int out = fh->pointHessiansMarginalized.size() + fh->pointHessiansOut.size();


        Vec2 refToFh = AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure,
                                                   frameHessians.back()->aff_g2l(), fh->aff_g2l());


        if( (in < setting_minPointsRemaining * (in + out) ||
             fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow)
            && ((int)frameHessians.size()) - flagged > setting_minFrames)
        {
//          LOG_INFO("MARGINALIZE frame %d, as only %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//                  fh->frameID, in, in+out,
//                  (int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//                  (int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//                  visInLast, outInLast,
//                  fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
            fh->flaggedForMarginalization = true;
            flagged++;
        }
        else
        {
//          LOG_INFO("May Keep frame %d, as %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//                  fh->frameID, in, in+out,
//                  (int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//                  (int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//                  visInLast, outInLast,
//                  fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
        }
    }

    // marginalize one.
    if((int)frameHessians.size() - flagged >= setting_maxFrames)
    {
        double smallestScore = 1;
        std::shared_ptr<FrameHessian> toMarginalize = nullptr;
        std::shared_ptr<FrameHessian> latest = frameHessians.back();


        for(auto& fh : frameHessians)
        {
            if(fh->frameID > latest->frameID - setting_minFrameAge || fh->frameID == 0)
            {
                continue;
            }

            //if(fh==frameHessians.front() == 0) continue;

            double distScore = 0;

            for(FrameFramePrecalc& ffh : fh->targetPrecalc)
            {
                if(ffh.target->frameID > latest->frameID - setting_minFrameAge + 1 || ffh.target == ffh.host)
                {
                    continue;
                }

                distScore += 1 / (1e-5 + ffh.distanceLL);

            }

            distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);


            if(distScore < smallestScore)
            {
                smallestScore = distScore;
                toMarginalize = fh;
            }
        }

//      LOG_INFO("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
//              toMarginalize->frameID, smallestScore);
        toMarginalize->flaggedForMarginalization = true;
        flagged++;
    }

//  LOG_INFO("FRAMES LEFT: ");
//  for(FrameHessian* fh : frameHessians)
//      LOG_INFO("%d ", fh->frameID);
//  LOG_INFO("\n");
}




void FullSystem::marginalizeFrame(std::shared_ptr<FrameHessian> frame)
{
    // marginalize or remove all this frames points.

    assert((int)frame->pointHessians.size() == 0);


    ef->marginalizeFrame(frame->efFrame);

    // drop all observations of existing points in that frame.
    int posOfSame = -1;
    int i = 0;

    for(std::shared_ptr<FrameHessian> fh : frameHessians)
    {
        if(fh.get() == frame.get())
        {
            posOfSame = i;
            continue;
        }

        i++;

        for(PointHessian* ph : fh->pointHessians)
        {
            for(unsigned int i = 0; i < ph->residuals.size(); i++)
            {
                PointFrameResidual* r = ph->residuals[i];

                if(r->target == frame.get())
                {
                    if(ph->lastResiduals[0].first == r)
                    {
                        ph->lastResiduals[0].first = 0;
                    }
                    else if(ph->lastResiduals[1].first == r)
                    {
                        ph->lastResiduals[1].first = 0;
                    }


                    if(r->host->frameID < r->target->frameID)
                    {
                        statistics_numForceDroppedResFwd++;
                    }
                    else
                    {
                        statistics_numForceDroppedResBwd++;
                    }

                    ef->dropResidual(r->efResidual);
                    deleteOut<PointFrameResidual>(ph->residuals, i);
                    break;
                }
            }
        }
    }



    {
        std::vector<std::shared_ptr<FrameHessian>> v;
        v.push_back(frame);
        std::map<int, ::Viewer::KeyFrameView> view;

        if (!outputWrapper.empty())
        {
            updateFrameHessiansView(v, view, Hcalib);
        }

        for(auto& ow : outputWrapper)
        {
            ow->publishKeyframes(view, true);
        }
    }


    frame->shell->marginalizedAt = frameHessians.back()->shell->id;
    frame->shell->movedByOpt = frame->w2c_leftEps().norm();


//deleteOutOrder<std::shared_ptr<FrameHessian>>(frameHessians, frame);
    auto it = frameHessians.begin();

    if (posOfSame >= 0)
    {
        frameHessians.erase(it + posOfSame);
    }
    else
    {
        auto end = frameHessians.end();

        for (; it != end; ++it)
        {
            if (frame.get() == it->get())
            {
                frameHessians.erase(it);
                break;
            }
        }
    }

    for(unsigned int i = 0; i < frameHessians.size(); i++)
    {
        frameHessians[i]->idx = i;
    }

    setPrecalcValues();
    ef->setAdjointsF(&Hcalib);
}




}
