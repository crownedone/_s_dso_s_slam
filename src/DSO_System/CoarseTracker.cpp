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

#include "DSO_system/CoarseTracker.hpp"
#include "DSO_system/FullSystem.hpp"
#include "DSO_system/HessianBlocks.hpp"
#include "DSO_system/Residuals.hpp"
#include "DSO_system/ImmaturePoint.hpp"
#include <algorithm>
#include "OptimizationBackend/EnergyFunctionalStructs.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "OpenCL/OpenCLHelper.hpp"
#include "OpenCL/KERNELS.hpp"

#include <Eigen/Cholesky>
#include <Eigen/LU>
#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
    #include "SSE2NEON.hpp"
#endif

namespace dso
{

CoarseTracker::CoarseTracker(int ww, int hh) : lastRef_aff_g2l(0, 0)
{
    // make coarse tracking templates.
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        int wl = ww >> lvl;
        int hl = hh >> lvl;
        idepth[lvl] = new float[wl * hl];
        weightSums[lvl] = new float[wl * hl];
        weightSums_bak[lvl] = new float[wl * hl];


        pc_u[lvl] = new float[wl * hl];
        pc_v[lvl] = new float[wl * hl];
        pc_idepth[lvl] = new float[wl * hl];
        pc_color[lvl] = new float[wl * hl];

    }

    // warped buffers
    buf_warped_idepth = new float[ww * hh];
    buf_warped_u = new float[ww * hh];
    buf_warped_v = new float[ww * hh];
    buf_warped_dx = new float[ww * hh];
    buf_warped_dy = new float[ww * hh];
    buf_warped_residual = new float[ww * hh];
    buf_warped_weight = new float[ww * hh];
    buf_warped_refColor = new float[ww * hh];


    newFrame = 0;
    lastRef = 0;
    debugPlot = debugPrint = true;
    w[0] = h[0] = 0;
    refFrameID = -1;
}
CoarseTracker::~CoarseTracker()
{
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        delete[] idepth[lvl];
        delete[] weightSums[lvl];
        delete[] weightSums_bak[lvl];

        delete[] pc_u[lvl];
        delete[] pc_v[lvl];
        delete[] pc_idepth[lvl];
        delete[] pc_color[lvl];


    }

    delete[]  buf_warped_idepth;
    delete[]  buf_warped_u;
    delete[]  buf_warped_v;
    delete[]  buf_warped_dx;
    delete[]  buf_warped_dy;
    delete[]  buf_warped_residual;
    delete[]  buf_warped_weight;
    delete[]  buf_warped_refColor;

}

void CoarseTracker::makeK(CalibHessian* HCalib)
{
    w[0] = wG[0];
    h[0] = hG[0];

    fx[0] = HCalib->fxl();
    fy[0] = HCalib->fyl();
    cx[0] = HCalib->cxl();
    cy[0] = HCalib->cyl();

    for (int level = 1; level < pyrLevelsUsed; ++ level)
    {
        w[level] = w[0] >> level;
        h[level] = h[0] >> level;
        fx[level] = fx[level - 1] * 0.5f;
        fy[level] = fy[level - 1] * 0.5f;
        cx[level] = (cx[0] + 0.5f) / ((int)1 << level) - 0.5f;
        cy[level] = (cy[0] + 0.5f) / ((int)1 << level) - 0.5f;
    }

    for (int level = 0; level < pyrLevelsUsed; ++ level)
    {
        K[level]  << fx[level], 0.0f, cx[level], 0.0f, fy[level], cy[level], 0.0f, 0.0f, 1.0f;
        Ki[level] = K[level].inverse();
        fxi[level] = Ki[level](0, 0);
        fyi[level] = Ki[level](1, 1);
        cxi[level] = Ki[level](0, 2);
        cyi[level] = Ki[level](1, 2);
    }
}


void CoarseTracker::makeCoarseDepthForFirstFrame(std::shared_ptr<FrameHessian> fh)
{
    // make coarse tracking templates for latstRef.
    memset(idepth[0], 0, sizeof(float) * w[0] * h[0]);
    memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);

    for (PointHessian* ph : fh->pointHessians)
    {
        int u = ph->u + 0.5f;
        int v = ph->v + 0.5f;
        float new_idepth = ph->idepth;
        float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));
        idepth[0][u + w[0] * v] += new_idepth * weight;
        weightSums[0][u + w[0] * v] += weight;

    }

    for (int lvl = 1; lvl < pyrLevelsUsed; lvl++)
    {
        int lvlm1 = lvl - 1;
        int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

        float* idepth_l = idepth[lvl];
        float* weightSums_l = weightSums[lvl];

        float* idepth_lm = idepth[lvlm1];
        float* weightSums_lm = weightSums[lvlm1];

        for (int y = 0; y < hl; y++)
            for (int x = 0; x < wl; x++)
            {
                int bidx = 2 * x + 2 * y * wlm1;
                idepth_l[x + y * wl] = idepth_lm[bidx] +
                                       idepth_lm[bidx + 1] +
                                       idepth_lm[bidx + wlm1] +
                                       idepth_lm[bidx + wlm1 + 1];

                weightSums_l[x + y * wl] = weightSums_lm[bidx] +
                                           weightSums_lm[bidx + 1] +
                                           weightSums_lm[bidx + wlm1] +
                                           weightSums_lm[bidx + wlm1 + 1];
            }
    }

    // dilate idepth by 1.
    for (int lvl = 0; lvl < 2; lvl++)
    {
        int numIts = 1;


        for (int it = 0; it < numIts; it++)
        {
            int wh = w[lvl] * h[lvl] - w[lvl];
            int wl = w[lvl];
            float* weightSumsl = weightSums[lvl];
            float* weightSumsl_bak = weightSums_bak[lvl];
            memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
            float* idepthl = idepth[lvl];   // dont need to make a temp copy of depth, since I only

            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for (int i = w[lvl]; i < wh; i++)
            {
                if (weightSumsl_bak[i] <= 0)
                {
                    float sum = 0, num = 0, numn = 0;

                    if (weightSumsl_bak[i + 1 + wl] > 0)
                    {
                        sum += idepthl[i + 1 + wl];
                        num += weightSumsl_bak[i + 1 + wl];
                        numn++;
                    }

                    if (weightSumsl_bak[i - 1 - wl] > 0)
                    {
                        sum += idepthl[i - 1 - wl];
                        num += weightSumsl_bak[i - 1 - wl];
                        numn++;
                    }

                    if (weightSumsl_bak[i + wl - 1] > 0)
                    {
                        sum += idepthl[i + wl - 1];
                        num += weightSumsl_bak[i + wl - 1];
                        numn++;
                    }

                    if (weightSumsl_bak[i - wl + 1] > 0)
                    {
                        sum += idepthl[i - wl + 1];
                        num += weightSumsl_bak[i - wl + 1];
                        numn++;
                    }

                    if (numn > 0)
                    {
                        idepthl[i] = sum / numn;
                        weightSumsl[i] = num / numn;
                    }
                }
            }
        }
    }


    // dilate idepth by 1 (2 on lower levels).
    for (int lvl = 2; lvl < pyrLevelsUsed; lvl++)
    {
        int wh = w[lvl] * h[lvl] - w[lvl];
        int wl = w[lvl];
        float* weightSumsl = weightSums[lvl];
        float* weightSumsl_bak = weightSums_bak[lvl];
        memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
        float* idepthl = idepth[lvl];   // dotnt need to make a temp copy of depth, since I only

        // read values with weightSumsl>0, and write ones with weightSumsl<=0.
        for (int i = w[lvl]; i < wh; i++)
        {
            if (weightSumsl_bak[i] <= 0)
            {
                float sum = 0, num = 0, numn = 0;

                if (weightSumsl_bak[i + 1] > 0)
                {
                    sum += idepthl[i + 1];
                    num += weightSumsl_bak[i + 1];
                    numn++;
                }

                if (weightSumsl_bak[i - 1] > 0)
                {
                    sum += idepthl[i - 1];
                    num += weightSumsl_bak[i - 1];
                    numn++;
                }

                if (weightSumsl_bak[i + wl] > 0)
                {
                    sum += idepthl[i + wl];
                    num += weightSumsl_bak[i + wl];
                    numn++;
                }

                if (weightSumsl_bak[i - wl] > 0)
                {
                    sum += idepthl[i - wl];
                    num += weightSumsl_bak[i - wl];
                    numn++;
                }

                if (numn > 0)
                {
                    idepthl[i] = sum / numn;
                    weightSumsl[i] = num / numn;
                }
            }
        }
    }


    // normalize idepths and weights.
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        float* weightSumsl = weightSums[lvl];
        float* idepthl = idepth[lvl];
        Eigen::Vector3f* dIRefl = lastRef->dIp[lvl].ptr<Eigen::Vector3f>();

        int wl = w[lvl], hl = h[lvl];

        int lpc_n = 0;
        float* lpc_u = pc_u[lvl];
        float* lpc_v = pc_v[lvl];
        float* lpc_idepth = pc_idepth[lvl];
        float* lpc_color = pc_color[lvl];


        for (int y = 2; y < hl - 2; y++)
            for (int x = 2; x < wl - 2; x++)
            {
                int i = x + y * wl;

                if (weightSumsl[i] > 0)
                {
                    idepthl[i] /= weightSumsl[i];
                    lpc_u[lpc_n] = x;
                    lpc_v[lpc_n] = y;
                    lpc_idepth[lpc_n] = idepthl[i];
                    lpc_color[lpc_n] = dIRefl[i][0];



                    if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0))
                    {
                        idepthl[i] = -1;
                        continue;   // just skip if something is wrong.
                    }

                    lpc_n++;
                }
                else
                {
                    idepthl[i] = -1;
                }

                weightSumsl[i] = 1;
            }

        pc_n[lvl] = lpc_n;
        //      printf("pc_n[lvl] is %d \n", lpc_n);
    }

}

void CoarseTracker::makeCoarseDepthL0(std::vector<std::shared_ptr<FrameHessian>> frameHessians)
{
    // make coarse tracking templates for latstRef.
    memset(idepth[0], 0, sizeof(float) * w[0] * h[0]);
    memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);

    for (auto fh : frameHessians)
    {
        for (auto ph : fh->pointHessians)
        {
            if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::INP)
            {
                PointFrameResidual* r = ph->lastResiduals[0].first;
                assert(r->efResidual->isActive() && r->target == lastRef.get());
                int u = r->centerProjectedTo[0] + 0.5f;
                int v = r->centerProjectedTo[1] + 0.5f;
                float new_idepth = r->centerProjectedTo[2];
                float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));

                idepth[0][u + w[0] * v] += new_idepth * weight;
                weightSums[0][u + w[0] * v] += weight;
            }
        }
    }


    for (int lvl = 1; lvl < pyrLevelsUsed; lvl++)
    {
        int lvlm1 = lvl - 1;
        int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

        float* idepth_l = idepth[lvl];
        float* weightSums_l = weightSums[lvl];

        float* idepth_lm = idepth[lvlm1];
        float* weightSums_lm = weightSums[lvlm1];

        for (int y = 0; y < hl; y++)
            for (int x = 0; x < wl; x++)
            {
                int bidx = 2 * x + 2 * y * wlm1;
                idepth_l[x + y * wl] = idepth_lm[bidx] +
                                       idepth_lm[bidx + 1] +
                                       idepth_lm[bidx + wlm1] +
                                       idepth_lm[bidx + wlm1 + 1];

                weightSums_l[x + y * wl] = weightSums_lm[bidx] +
                                           weightSums_lm[bidx + 1] +
                                           weightSums_lm[bidx + wlm1] +
                                           weightSums_lm[bidx + wlm1 + 1];
            }
    }


    // dilate idepth by 1.
    for (int lvl = 0; lvl < 2; lvl++)
    {
        int numIts = 1;


        for (int it = 0; it < numIts; it++)
        {
            int wh = w[lvl] * h[lvl] - w[lvl];
            int wl = w[lvl];
            float* weightSumsl = weightSums[lvl];
            float* weightSumsl_bak = weightSums_bak[lvl];
            memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
            float* idepthl = idepth[lvl];   // dotnt need to make a temp copy of depth, since I only

            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for (int i = w[lvl]; i < wh; i++)
            {
                if (weightSumsl_bak[i] <= 0)
                {
                    float sum = 0, num = 0, numn = 0;

                    if (weightSumsl_bak[i + 1 + wl] > 0)
                    {
                        sum += idepthl[i + 1 + wl];
                        num += weightSumsl_bak[i + 1 + wl];
                        numn++;
                    }

                    if (weightSumsl_bak[i - 1 - wl] > 0)
                    {
                        sum += idepthl[i - 1 - wl];
                        num += weightSumsl_bak[i - 1 - wl];
                        numn++;
                    }

                    if (weightSumsl_bak[i + wl - 1] > 0)
                    {
                        sum += idepthl[i + wl - 1];
                        num += weightSumsl_bak[i + wl - 1];
                        numn++;
                    }

                    if (weightSumsl_bak[i - wl + 1] > 0)
                    {
                        sum += idepthl[i - wl + 1];
                        num += weightSumsl_bak[i - wl + 1];
                        numn++;
                    }

                    if (numn > 0)
                    {
                        idepthl[i] = sum / numn;
                        weightSumsl[i] = num / numn;
                    }
                }
            }
        }
    }


    // dilate idepth by 1 (2 on lower levels).
    for (int lvl = 2; lvl < pyrLevelsUsed; lvl++)
    {
        int wh = w[lvl] * h[lvl] - w[lvl];
        int wl = w[lvl];
        float* weightSumsl = weightSums[lvl];
        float* weightSumsl_bak = weightSums_bak[lvl];
        memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
        float* idepthl = idepth[lvl];   // dotnt need to make a temp copy of depth, since I only

        // read values with weightSumsl>0, and write ones with weightSumsl<=0.
        for (int i = w[lvl]; i < wh; i++)
        {
            if (weightSumsl_bak[i] <= 0)
            {
                float sum = 0, num = 0, numn = 0;

                if (weightSumsl_bak[i + 1] > 0)
                {
                    sum += idepthl[i + 1];
                    num += weightSumsl_bak[i + 1];
                    numn++;
                }

                if (weightSumsl_bak[i - 1] > 0)
                {
                    sum += idepthl[i - 1];
                    num += weightSumsl_bak[i - 1];
                    numn++;
                }

                if (weightSumsl_bak[i + wl] > 0)
                {
                    sum += idepthl[i + wl];
                    num += weightSumsl_bak[i + wl];
                    numn++;
                }

                if (weightSumsl_bak[i - wl] > 0)
                {
                    sum += idepthl[i - wl];
                    num += weightSumsl_bak[i - wl];
                    numn++;
                }

                if (numn > 0)
                {
                    idepthl[i] = sum / numn;
                    weightSumsl[i] = num / numn;
                }
            }
        }
    }


    // normalize idepths and weights.
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        float* weightSumsl = weightSums[lvl];
        float* idepthl = idepth[lvl];
        Eigen::Vector3f* dIRefl = lastRef->dIp[lvl].ptr<Eigen::Vector3f>();

        int wl = w[lvl], hl = h[lvl];

        int lpc_n = 0;
        float* lpc_u = pc_u[lvl];
        float* lpc_v = pc_v[lvl];
        float* lpc_idepth = pc_idepth[lvl];
        float* lpc_color = pc_color[lvl];


        for (int y = 2; y < hl - 2; y++)
            for (int x = 2; x < wl - 2; x++)
            {
                int i = x + y * wl;

                if (weightSumsl[i] > 0)
                {
                    idepthl[i] /= weightSumsl[i];
                    lpc_u[lpc_n] = x;
                    lpc_v[lpc_n] = y;
                    lpc_idepth[lpc_n] = idepthl[i];
                    lpc_color[lpc_n] = dIRefl[i][0];



                    if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0))
                    {
                        idepthl[i] = -1;
                        continue;   // just skip if something is wrong.
                    }

                    lpc_n++;
                }
                else
                {
                    idepthl[i] = -1;
                }

                weightSumsl[i] = 1;
            }

        pc_n[lvl] = lpc_n;
    }

}

// make depth mainly from static stereo matching and fill the holes from propogation idpeth map.
void CoarseTracker::makeCoarseDepthL0(std::vector<std::shared_ptr<FrameHessian>> frameHessians,
                                      std::shared_ptr<FrameHessian> fh_right, CalibHessian Hcalib)
{
    // make coarse tracking templates for latstRef.
    memset(idepth[0], 0, sizeof(float) * w[0] * h[0]);
    memset(weightSums[0], 0, sizeof(float) * w[0] * h[0]);

    std::shared_ptr<FrameHessian> fh_target = frameHessians.back();
    Mat33f K1 = Mat33f::Identity();
    K1(0, 0) = Hcalib.fxl();
    K1(1, 1) = Hcalib.fyl();
    K1(0, 2) = Hcalib.cxl();
    K1(1, 2) = Hcalib.cyl();

    for (auto& fh : frameHessians)
    {
        for (PointHessian* ph : fh->pointHessians)
        {
            if (ph->lastResiduals[0].first != 0 &&
                ph->lastResiduals[0].second ==
                ResState::INP) //contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).
            {
                PointFrameResidual* r = ph->lastResiduals[0].first;
                assert(r->efResidual->isActive() && r->target == lastRef.get());
                int u = r->centerProjectedTo[0] + 0.5f;
                int v = r->centerProjectedTo[1] + 0.5f;

                ImmaturePoint* pt_track = new ImmaturePoint((float)u, (float)v, fh_target.get(), &Hcalib);

                pt_track->u_stereo = pt_track->u;
                pt_track->v_stereo = pt_track->v;

                // free to debug
                pt_track->idepth_min_stereo = r->centerProjectedTo[2] * 0.1f;
                pt_track->idepth_max_stereo = r->centerProjectedTo[2] * 1.9f;

                ImmaturePointStatus pt_track_right = pt_track->traceStereo(fh_right->dI_ptr, K1, 1);

                float new_idepth = 0;

                if (pt_track_right == ImmaturePointStatus::IPS_GOOD)
                {
                    ImmaturePoint* pt_track_back = new ImmaturePoint(pt_track->lastTraceUV(0), pt_track->lastTraceUV(1), fh_right.get(), &Hcalib);
                    pt_track_back->u_stereo = pt_track_back->u;
                    pt_track_back->v_stereo = pt_track_back->v;


                    pt_track_back->idepth_min_stereo = r->centerProjectedTo[2] * 0.1f;
                    pt_track_back->idepth_max_stereo = r->centerProjectedTo[2] * 1.9f;

                    ImmaturePointStatus pt_track_left = pt_track_back->traceStereo(fh_target->dI_ptr, K1, 0);

                    float depth = 1.0f / pt_track->idepth_stereo;
                    float u_delta = abs(pt_track->u - pt_track_back->lastTraceUV(0));

                    if (u_delta < 1 && depth > 0 && depth < 50)
                    {
                        new_idepth = pt_track->idepth_stereo;
                        delete pt_track;
                        delete pt_track_back;

                    }
                    else
                    {

                        new_idepth = r->centerProjectedTo[2];
                        delete pt_track;
                        delete pt_track_back;
                    }

                }
                else
                {

                    new_idepth = r->centerProjectedTo[2];
                    delete pt_track;

                }

                float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));

                idepth[0][u + w[0] * v] += new_idepth * weight;
                weightSums[0][u + w[0] * v] += weight;

            }
        }
    }

    for (int lvl = 1; lvl < pyrLevelsUsed; lvl++)
    {
        int lvlm1 = lvl - 1;
        int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

        float* idepth_l = idepth[lvl];
        float* weightSums_l = weightSums[lvl];

        float* idepth_lm = idepth[lvlm1];
        float* weightSums_lm = weightSums[lvlm1];

        for (int y = 0; y < hl; y++)
            for (int x = 0; x < wl; x++)
            {
                int bidx = 2 * x + 2 * y * wlm1;
                idepth_l[x + y * wl] = idepth_lm[bidx] +
                                       idepth_lm[bidx + 1] +
                                       idepth_lm[bidx + wlm1] +
                                       idepth_lm[bidx + wlm1 + 1];

                weightSums_l[x + y * wl] = weightSums_lm[bidx] +
                                           weightSums_lm[bidx + 1] +
                                           weightSums_lm[bidx + wlm1] +
                                           weightSums_lm[bidx + wlm1 + 1];
            }
    }

    // dilate idepth by 1.
    for (int lvl = 0; lvl < 2; lvl++)
    {
        int numIts = 1;


        for (int it = 0; it < numIts; it++)
        {
            int wh = w[lvl] * h[lvl] - w[lvl];
            int wl = w[lvl];
            float* weightSumsl = weightSums[lvl];
            float* weightSumsl_bak = weightSums_bak[lvl];
            memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
            float* idepthl = idepth[lvl];   // dont need to make a temp copy of depth, since I only

            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for (int i = w[lvl]; i < wh; i++)
            {
                if (weightSumsl_bak[i] <= 0)
                {
                    float sum = 0, num = 0, numn = 0;

                    if (weightSumsl_bak[i + 1 + wl] > 0)
                    {
                        sum += idepthl[i + 1 + wl];
                        num += weightSumsl_bak[i + 1 + wl];
                        numn++;
                    }

                    if (weightSumsl_bak[i - 1 - wl] > 0)
                    {
                        sum += idepthl[i - 1 - wl];
                        num += weightSumsl_bak[i - 1 - wl];
                        numn++;
                    }

                    if (weightSumsl_bak[i + wl - 1] > 0)
                    {
                        sum += idepthl[i + wl - 1];
                        num += weightSumsl_bak[i + wl - 1];
                        numn++;
                    }

                    if (weightSumsl_bak[i - wl + 1] > 0)
                    {
                        sum += idepthl[i - wl + 1];
                        num += weightSumsl_bak[i - wl + 1];
                        numn++;
                    }

                    if (numn > 0)
                    {
                        idepthl[i] = sum / numn;
                        weightSumsl[i] = num / numn;
                    }
                }
            }
        }
    }


    // dilate idepth by 1 (2 on lower levels).
    for (int lvl = 2; lvl < pyrLevelsUsed; lvl++)
    {
        int wh = w[lvl] * h[lvl] - w[lvl];
        int wl = w[lvl];
        float* weightSumsl = weightSums[lvl];
        float* weightSumsl_bak = weightSums_bak[lvl];
        memcpy(weightSumsl_bak, weightSumsl, w[lvl] * h[lvl] * sizeof(float));
        float* idepthl = idepth[lvl];   // dotnt need to make a temp copy of depth, since I only

        // read values with weightSumsl>0, and write ones with weightSumsl<=0.
        for (int i = w[lvl]; i < wh; i++)
        {
            if (weightSumsl_bak[i] <= 0)
            {
                float sum = 0, num = 0, numn = 0;

                if (weightSumsl_bak[i + 1] > 0)
                {
                    sum += idepthl[i + 1];
                    num += weightSumsl_bak[i + 1];
                    numn++;
                }

                if (weightSumsl_bak[i - 1] > 0)
                {
                    sum += idepthl[i - 1];
                    num += weightSumsl_bak[i - 1];
                    numn++;
                }

                if (weightSumsl_bak[i + wl] > 0)
                {
                    sum += idepthl[i + wl];
                    num += weightSumsl_bak[i + wl];
                    numn++;
                }

                if (weightSumsl_bak[i - wl] > 0)
                {
                    sum += idepthl[i - wl];
                    num += weightSumsl_bak[i - wl];
                    numn++;
                }

                if (numn > 0)
                {
                    idepthl[i] = sum / numn;
                    weightSumsl[i] = num / numn;
                }
            }
        }
    }


    // normalize idepths and weights.
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        float* weightSumsl = weightSums[lvl];
        float* idepthl = idepth[lvl];
        Eigen::Vector3f* dIRefl = lastRef->dIp[lvl].ptr<Eigen::Vector3f>();

        int wl = w[lvl], hl = h[lvl];

        int lpc_n = 0;
        float* lpc_u = pc_u[lvl];
        float* lpc_v = pc_v[lvl];
        float* lpc_idepth = pc_idepth[lvl];
        float* lpc_color = pc_color[lvl];


        for (int y = 2; y < hl - 2; y++)
            for (int x = 2; x < wl - 2; x++)
            {
                int i = x + y * wl;

                if (weightSumsl[i] > 0)
                {
                    idepthl[i] /= weightSumsl[i];
                    lpc_u[lpc_n] = x;
                    lpc_v[lpc_n] = y;
                    lpc_idepth[lpc_n] = idepthl[i];
                    lpc_color[lpc_n] = dIRefl[i][0];



                    if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0))
                    {
                        idepthl[i] = -1;
                        continue;   // just skip if something is wrong.
                    }

                    lpc_n++;
                }
                else
                {
                    idepthl[i] = -1;
                }

                weightSumsl[i] = 1;
            }

        pc_n[lvl] = lpc_n;
    }

}


void CoarseTracker::calcGSSSE(int lvl, Mat88& H_out, Vec8& b_out, const SE3& refToNew,
                              AffLight aff_g2l)
{
    acc.initialize();

    __m128 fxl = _mm_set1_ps(fx[lvl]);
    __m128 fyl = _mm_set1_ps(fy[lvl]);
    __m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
    __m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure,
                                   newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));

    __m128 one = _mm_set1_ps(1);
    __m128 minusOne = _mm_set1_ps(-1);
    __m128 zero = _mm_set1_ps(0);

    int n = buf_warped_n;
    assert(n % 4 == 0);

    for(int i = 0; i < n; i += 4)
    {
        __m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx + i), fxl);
        __m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy + i), fyl);
        __m128 u = _mm_load_ps(buf_warped_u + i);
        __m128 v = _mm_load_ps(buf_warped_v + i);
        __m128 id = _mm_load_ps(buf_warped_idepth + i);


        acc.updateSSE_eighted(
            _mm_mul_ps(id, dx),
            _mm_mul_ps(id, dy),
            _mm_sub_ps(zero, _mm_mul_ps(id, _mm_add_ps(_mm_mul_ps(u, dx), _mm_mul_ps(v, dy)))),
            _mm_sub_ps(zero, _mm_add_ps(
                           _mm_mul_ps(_mm_mul_ps(u, v), dx),
                           _mm_mul_ps(dy, _mm_add_ps(one, _mm_mul_ps(v, v))))),
            _mm_add_ps(
                _mm_mul_ps(_mm_mul_ps(u, v), dy),
                _mm_mul_ps(dx, _mm_add_ps(one, _mm_mul_ps(u, u)))),
            _mm_sub_ps(_mm_mul_ps(u, dy), _mm_mul_ps(v, dx)),
            _mm_mul_ps(a, _mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor + i))),
            minusOne,
            _mm_load_ps(buf_warped_residual + i),
            _mm_load_ps(buf_warped_weight + i));
    }

    acc.finish();
    H_out = acc.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
    b_out = acc.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

    H_out.block<8, 3>(0, 0) *= SCALE_XI_ROT;
    H_out.block<8, 3>(0, 3) *= SCALE_XI_TRANS;
    H_out.block<8, 1>(0, 6) *= SCALE_A;
    H_out.block<8, 1>(0, 7) *= SCALE_B;
    H_out.block<3, 8>(0, 0) *= SCALE_XI_ROT;
    H_out.block<3, 8>(3, 0) *= SCALE_XI_TRANS;
    H_out.block<1, 8>(6, 0) *= SCALE_A;
    H_out.block<1, 8>(7, 0) *= SCALE_B;
    b_out.segment<3>(0) *= SCALE_XI_ROT;
    b_out.segment<3>(3) *= SCALE_XI_TRANS;
    b_out.segment<1>(6) *= SCALE_A;
    b_out.segment<1>(7) *= SCALE_B;
}


void residualKernel(int kWidth, float* in_id, float* in_x, float* in_y, Vec9f RKi, Vec3f t, Vec2f affLL, Vec4f fxfycxcy, int lvl,
                    Vec9f Ki, int wl, int hl, float* lpc_color, float* dINewl, float cutoffTH, float maxEnergy,
                    int* numTermsInWarped, // unique counter
                    float* buf_warped_idepth, float* buf_warped_u, float* buf_warped_v, float* buf_warped_dx, float* buf_warped_dy,
                    float* buf_warped_residual,
                    float* buf_warped_weight, float* buf_warped_refColor, Vec6* rs) // Results
{
    for (int i = 0; i < kWidth; ++i)
    {
        float id = in_id[i];
        float x = in_x[i];
        float y = in_y[i];

        Vec3f pt;
        pt <<
           RKi[0] * x + RKi[1] * y + RKi[2] * 1 + t[0] * id,
               RKi[3] * x + RKi[4] * y + RKi[5] * 1 + t[1] * id,
               RKi[6] * x + RKi[7] * y + RKi[8] * 1 + t[2] * id;

        float u = pt[0] / pt[2];
        float v = pt[1] / pt[2];
        float Ku = fxfycxcy[0] * u + fxfycxcy[2];
        float Kv = fxfycxcy[1] * v + fxfycxcy[3];
        float new_idepth = id / pt[2];
        //LOG_INFO("Ku & Kv are: %f, %f; x and y are: %f, %f", Ku, Kv, x, y);

        if (lvl == 0 && i % 32 == 0)
        {
            // translation only (positive)
            Vec3f ptT;
            ptT << Ki[0] * x + Ki[1] * y + Ki[2] * 1 + t[0] * id,
                Ki[3] * x + Ki[4] * y + Ki[5] * 1 + t[1] * id,
                Ki[6] * x + Ki[7] * y + Ki[8] * 1 + t[2] * id;

            float uT = ptT[0] / ptT[2];
            float vT = ptT[1] / ptT[2];
            float KuT = fxfycxcy[0] * uT + fxfycxcy[2];
            float KvT = fxfycxcy[1] * vT + fxfycxcy[3];

            // translation only (negative)
            Vec3f ptT2;
            ptT2 << Ki[0] * x + Ki[1] * y + Ki[2] * 1 - t[0] * id,
                 Ki[3] * x + Ki[4] * y + Ki[5] * 1 - t[1] * id,
                 Ki[6] * x + Ki[7] * y + Ki[8] * 1 - t[2] * id;

            float uT2 = ptT2[0] / ptT2[2];
            float vT2 = ptT2[1] / ptT2[2];
            float KuT2 = fxfycxcy[0] * uT2 + fxfycxcy[2];
            float KvT2 = fxfycxcy[1] * vT2 + fxfycxcy[3];

            //translation and rotation (negative)
            Vec3f pt3;
            pt3 << RKi[0] * x + RKi[1] * y + RKi[2] * 1 - t[0] * id,
                RKi[3] * x + RKi[4] * y + RKi[5] * 1 - t[1] * id,
                RKi[6] * x + RKi[7] * y + RKi[8] * 1 - t[2] * id;

            float u3 = pt3[0] / pt3[2];
            float v3 = pt3[1] / pt3[2];
            float Ku3 = fxfycxcy[0] * u3 + fxfycxcy[2];
            float Kv3 = fxfycxcy[1] * v3 + fxfycxcy[3];

            //translation and rotation (positive)
            //already have it.
            (*rs)[2] += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
            (*rs)[2] += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
            (*rs)[4] += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
            (*rs)[4] += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
            (*rs)[3] += 2;
        }

        if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0))
        {
            continue;
        }

        float refColor = lpc_color[i];
        Vec3f hitColor;
        {
            int ix = (int)Ku;
            int iy = (int)Kv;
            float dx = Ku - ix;
            float dy = Kv - iy;
            float dxdy = dx * dy;
            const float* bp = dINewl + 3 * (ix + iy * wl);

            hitColor << dxdy * (*(bp + 3 * (1 + wl)))
                     + (dy - dxdy) * (*(bp + 3 * wl))
                     + (dx - dxdy) * (*(bp + 3))
                     + (1 - dx - dy + dxdy) * (*bp),
                     dxdy* (*(bp + 3 * (1 + wl) + 1))
                     + (dy - dxdy) * (*(bp + 3 * wl + 1))
                     + (dx - dxdy) * (*(bp + 3 + 1))
                     + (1 - dx - dy + dxdy) * (*(bp + 1)),
                     dxdy* (*(bp + 3 * (1 + wl) + 2))
                     + (dy - dxdy) * (*(bp + 3 * wl + 2))
                     + (dx - dxdy) * (*(bp + 3 + 2))
                     + (1 - dx - dy + dxdy) * (*(bp + 2));

            //LOG_INFO_IF(i == 0, "%f,%f,%f", hitColor[0], hitColor[1], hitColor[2]);
        }

        if (!std::isfinite((float)hitColor[0]))
        {
            continue;
        }

        float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
        //Huber weight
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


        if (fabs(residual) > cutoffTH)
        {
            (*rs)[0] += maxEnergy;
            (*rs)[1]++;
            (*rs)[5]++;
        }
        else
        {
            (*rs)[0] += hw * residual * residual * (2 - hw);
            (*rs)[1]++;

            buf_warped_idepth[*numTermsInWarped] = new_idepth;
            buf_warped_u[*numTermsInWarped] = u;
            buf_warped_v[*numTermsInWarped] = v;
            buf_warped_dx[*numTermsInWarped] = hitColor[1];
            buf_warped_dy[*numTermsInWarped] = hitColor[2];
            buf_warped_residual[*numTermsInWarped] = residual;
            buf_warped_weight[*numTermsInWarped] = hw;
            buf_warped_refColor[*numTermsInWarped] = lpc_color[i];
            (*numTermsInWarped)++;
        }
    }
}

cv::ocl::ProgramSource KERNEL_calcRes;
cv::UMat Uidepth, Ulpc_u, Ulpc_v, Ulpc_color;
cv::UMat UAtomicCounter;
cv::UMat Uwp, Urs;

Vec6 CoarseTracker::calcRes(int lvl, const SE3& refToNew, AffLight aff_g2l, float cutoffTH)
{
    debugPlot = false;


    int wl = w[lvl];
    int hl = h[lvl];
    float fxl = fx[lvl];
    float fyl = fy[lvl];
    float cxl = cx[lvl];
    float cyl = cy[lvl];

    Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
    //LOG_INFO("the Ki is:\n %f,%f,%f\n %f,%f,%f\n %f,%f,%f\n -----\n", Ki[lvl](0, 0), Ki[lvl](0, 1), Ki[lvl](0, 2), Ki[lvl](1, 0),
    //         Ki[lvl](1, 1), Ki[lvl](1, 2), Ki[lvl](2, 0), Ki[lvl](2, 1), Ki[lvl](2, 2) );
    Vec3f t = (refToNew.translation()).cast<float>();
    //LOG_INFO("the t is:\n %f, %f, %f\n", t(0), t(1), t(2));
    Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure,
                                              lastRef_aff_g2l, aff_g2l).cast<float>();


    float sumSquaredShiftT = 0;
    float sumSquaredShiftRT = 0;
    float sumSquaredShiftNum = 0;

    float maxEnergy = 2 * setting_huberTH * cutoffTH - setting_huberTH *
                      setting_huberTH; // energy for r=setting_coarseCutoffTH.


    cv::Mat resImage = debugPlot ? cv::Mat(hl, wl, CV_8UC3, cv::Scalar::all(255)) : cv::Mat();

    int nl = pc_n[lvl];
    float* lpc_u = pc_u[lvl];
    float* lpc_v = pc_v[lvl];
    float* lpc_idepth = pc_idepth[lvl];
    float* lpc_color = pc_color[lvl];

    //LOG_INFO("the num of the points is: %d", nl);

    Vec6 rs = Vec6::Zero();
    int numTermsInWarped = 0;

    if (true)
    {
        Vec9f RKi_;
        RKi_ << RKi(0, 0), RKi(0, 1), RKi(0, 2),
             RKi(1, 0), RKi(1, 1), RKi(1, 2),
             RKi(2, 0), RKi(2, 1), RKi(2, 2);
        Vec4f fxfycxcy;
        fxfycxcy << fxl, fyl, cxl, cyl;
        Vec9f Ki_;
        Ki_ << Ki[lvl](0, 0), Ki[lvl](0, 1), Ki[lvl](0, 2),
            Ki[lvl](1, 0), Ki[lvl](1, 1), Ki[lvl](1, 2),
            Ki[lvl](2, 0), Ki[lvl](2, 1), Ki[lvl](2, 2);

        float* dINewl = newFrame->dIp[lvl].ptr<float>(); // actually Vec3f
        residualKernel(nl, lpc_idepth, lpc_u, lpc_v, RKi_, t, affLL, fxfycxcy, lvl, Ki_, wl, hl, lpc_color,
                       dINewl, cutoffTH, maxEnergy, &numTermsInWarped, buf_warped_idepth, buf_warped_u, buf_warped_v,
                       buf_warped_dx, buf_warped_dy, buf_warped_residual, buf_warped_weight,
                       buf_warped_refColor, &rs);

        if (setting_UseOpenCL)
        {
            // First time create kernels
            if (!KERNEL_calcRes.getImpl())
            {
                KERNEL_calcRes = cv::ocl::ProgramSource("coarseTracker", "calcRes", OCLKernels::KernelCalcRes, "");
                assert(!KERNEL_calcRes.getImpl());
                UAtomicCounter = cv::UMat(1, 1, CV_32SC1, cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY);
                Uwp = cv::UMat(1, h[0] * w[0] * 8, CV_32F, cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY);
                Urs = cv::UMat(1, 6 * h[0] * w[0], CV_32F, cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY);
            }


            // init to zero:
            UAtomicCounter.setTo(cv::Scalar::all(0));
            Urs.setTo(cv::Scalar::all(0));

            int kwidth = nl;
            size_t nll = nl;
            // Upload buffers
            cv::Mat(1, hl * wl, CV_32F, lpc_idepth, cv::Mat::AUTO_STEP).copyTo(Uidepth);
            cv::Mat(1, hl * wl, CV_32F, lpc_u, cv::Mat::AUTO_STEP).copyTo(Ulpc_u);
            cv::Mat(1, hl * wl, CV_32F, lpc_v, cv::Mat::AUTO_STEP).copyTo(Ulpc_v);
            cv::Mat(1, hl * wl, CV_32F, lpc_color, cv::Mat::AUTO_STEP).copyTo(Ulpc_color);

            std::array<float, 16> RKI_t_fxfycxcy =
            {
                RKi(0, 0), RKi(0, 1), RKi(0, 2), t[0],
                RKi(1, 0), RKi(1, 1), RKi(1, 2), t[1],
                RKi(2, 0), RKi(2, 1), RKi(2, 2), t[2],
                fxl, fyl, cxl, cyl
            };
            std::array<float, 16> Ki_affL_cutoffTH_maxEnergy =
            {
                Ki[lvl](0, 0), Ki[lvl](0, 1), Ki[lvl](0, 2), affLL[0],
                Ki[lvl](1, 0), Ki[lvl](1, 1), Ki[lvl](1, 2), affLL[1],
                Ki[lvl](2, 0), Ki[lvl](2, 1), Ki[lvl](2, 2), 0.f,
                cutoffTH, maxEnergy, setting_huberTH, 0.f
            };
            std::array<int, 4> kWidth_wl_hl_lvl =
            {
                nl, wl, hl, lvl
            };
            ocl::RunKernel("calcRes", KERNEL_calcRes,
            {
                cv::ocl::KernelArg::PtrReadOnly(Uidepth),
                cv::ocl::KernelArg::PtrReadOnly(Ulpc_u),
                cv::ocl::KernelArg::PtrReadOnly(Ulpc_v),
                cv::ocl::KernelArg::Constant(&RKI_t_fxfycxcy, sizeof(float) * 16),
                cv::ocl::KernelArg::Constant(&Ki_affL_cutoffTH_maxEnergy, sizeof(float) * 16),
                cv::ocl::KernelArg::Constant(&kWidth_wl_hl_lvl, sizeof(float) * 4),
                cv::ocl::KernelArg::PtrReadOnly(Ulpc_color),
                cv::ocl::KernelArg::PtrReadOnly(newFrame->dIp[lvl].getUMat(cv::AccessFlag::ACCESS_READ)),
                cv::ocl::KernelArg::PtrReadWrite(UAtomicCounter),
                cv::ocl::KernelArg::PtrWriteOnly(Uwp),
                cv::ocl::KernelArg::PtrReadWrite(Urs)
            }, { nll }, 1);

            // download buffers:
            cv::Mat aCounter;
            UAtomicCounter.copyTo(aCounter);
            numTermsInWarped = *(aCounter.ptr<int>());
            cv::Mat warped;
            // copy only the range we want.
            Uwp(cv::Range(0, 1), cv::Range(0, numTermsInWarped)).copyTo(warped);
            float* warpedPtr = warped.ptr<float>();

            for (int i = 0; i < numTermsInWarped; ++i)
            {
                buf_warped_idepth[i] = warpedPtr[8 * i + 0];
                buf_warped_u[i] = warpedPtr[8 * i + 1];
                buf_warped_v[i] = warpedPtr[8 * i + 2];
                buf_warped_dx[i] = warpedPtr[8 * i + 3];
                buf_warped_dy[i] = warpedPtr[8 * i + 4];
                buf_warped_residual[i] = warpedPtr[8 * i + 5];
                buf_warped_weight[i] = warpedPtr[8 * i + 6];
                buf_warped_refColor[i] = warpedPtr[8 * i + 7];
            }

            cv::Mat rsMat;
            Urs(cv::Range(0, 1), cv::Range(0, nl * 6)).copyTo(rsMat);
            float* rsMatPtr = rsMat.ptr<float>();

            for (int i = 0; i < nl; ++i)
            {
                rs[0] += rsMatPtr[i * 6 + 0];
                rs[1] += rsMatPtr[i * 6 + 1];
                rs[2] += rsMatPtr[i * 6 + 2];
                rs[3] += rsMatPtr[i * 6 + 3];
                rs[4] += rsMatPtr[i * 6 + 4];
                rs[5] += rsMatPtr[i * 6 + 5];
            }
        }




        // Global stuff
        // We temporarily misused some entries
        rs[2] /= (rs[3] + 0.1);
        rs[4] /= (rs[3] + 0.1);
        rs[3] = 0;
        rs[5] /= (float)(rs[1]);
    }
    else
    {
        Eigen::Vector3f* dINewl = newFrame->dIp[lvl].ptr<Eigen::Vector3f>();
        float E = 0;
        int numTermsInE = 0;
        int numSaturated = 0;

        for(int i = 0; i < nl; i++)
        {
            float id = lpc_idepth[i];
            float x = lpc_u[i];
            float y = lpc_v[i];

            Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
            float u = pt[0] / pt[2];
            float v = pt[1] / pt[2];
            float Ku = fxl * u + cxl;
            float Kv = fyl * v + cyl;
            float new_idepth = id / pt[2];
            //LOG_INFO("Ku & Kv are: %f, %f; x and y are: %f, %f", Ku, Kv, x, y);

            if(lvl == 0 && i % 32 == 0)
            {
                // translation only (positive)
                Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t * id;
                float uT = ptT[0] / ptT[2];
                float vT = ptT[1] / ptT[2];
                float KuT = fxl * uT + cxl;
                float KvT = fyl * vT + cyl;

                // translation only (negative)
                Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t * id;
                float uT2 = ptT2[0] / ptT2[2];
                float vT2 = ptT2[1] / ptT2[2];
                float KuT2 = fxl * uT2 + cxl;
                float KvT2 = fyl * vT2 + cyl;

                //translation and rotation (negative)
                Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
                float u3 = pt3[0] / pt3[2];
                float v3 = pt3[1] / pt3[2];
                float Ku3 = fxl * u3 + cxl;
                float Kv3 = fyl * v3 + cyl;

                //translation and rotation (positive)
                //already have it.

                sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
                sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
                sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
                sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
                sumSquaredShiftNum += 2;
            }

            if(!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0))
            {
                continue;
            }



            float refColor = lpc_color[i];
            Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
            //LOG_INFO_IF(i == 0, "%f,%f,%f", hitColor[0], hitColor[1], hitColor[2]);

            if(!std::isfinite((float)hitColor[0]))
            {
                continue;
            }

            float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
            //Huber weight
            float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);


            if(fabs(residual) > cutoffTH)
            {
                if(debugPlot)
                {
                    cv::circle(resImage, cv::Point(lpc_u[i], lpc_v[i]), 2, cv::Scalar(0, 0, 255));
                }

                E += maxEnergy;
                numTermsInE++;
                numSaturated++;
            }
            else
            {
                if(debugPlot)
                {
                    cv::circle(resImage, cv::Point(lpc_u[i], lpc_v[i]), 2,
                               cv::Scalar(residual + 128, residual + 128, residual + 128));
                }

                E += hw * residual * residual * (2 - hw);
                numTermsInE++;

                buf_warped_idepth[numTermsInWarped] = new_idepth;
                buf_warped_u[numTermsInWarped] = u;
                buf_warped_v[numTermsInWarped] = v;
                buf_warped_dx[numTermsInWarped] = hitColor[1];
                buf_warped_dy[numTermsInWarped] = hitColor[2];
                buf_warped_residual[numTermsInWarped] = residual;
                buf_warped_weight[numTermsInWarped] = hw;
                buf_warped_refColor[numTermsInWarped] = lpc_color[i];
                numTermsInWarped++;
            }
        }

        rs[0] = E;
        rs[1] = numTermsInE;
        rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);
        rs[3] = 0;
        rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);
        rs[5] = numSaturated / (float)numTermsInE;
    }


    while(numTermsInWarped % 4 != 0)
    {
        buf_warped_idepth[numTermsInWarped] = 0;
        buf_warped_u[numTermsInWarped] = 0;
        buf_warped_v[numTermsInWarped] = 0;
        buf_warped_dx[numTermsInWarped] = 0;
        buf_warped_dy[numTermsInWarped] = 0;
        buf_warped_residual[numTermsInWarped] = 0;
        buf_warped_weight[numTermsInWarped] = 0;
        buf_warped_refColor[numTermsInWarped] = 0;
        numTermsInWarped++;
    }

    buf_warped_n = numTermsInWarped;


    if(debugPlot)
    {
        Viewer::displayImage("RES", resImage, false);
        Viewer::waitKey(1);
    }

    LOG_INFO("Residual: %f, %f, %f, %f, %f, %f", rs[0], rs[1], rs[2], rs[3], rs[4], rs[5]);
    return rs;
}

void CoarseTracker::setCTRefForFirstFrame(std::vector<std::shared_ptr<FrameHessian>> frameHessians)
{
    assert(frameHessians.size() > 0);
    lastRef = frameHessians.back();

    makeCoarseDepthForFirstFrame(lastRef);

    refFrameID = lastRef->shell->id;
    lastRef_aff_g2l = lastRef->aff_g2l();

    firstCoarseRMSE = -1;
}

void CoarseTracker::setCoarseTrackingRef(
    std::vector<std::shared_ptr<FrameHessian>> frameHessians)
{
    assert(frameHessians.size() > 0);
    lastRef = frameHessians.back();
    makeCoarseDepthL0(frameHessians);



    refFrameID = lastRef->shell->id;
    lastRef_aff_g2l = lastRef->aff_g2l();

    firstCoarseRMSE = -1;

}
void CoarseTracker::setCoarseTrackingRef(std::vector<std::shared_ptr<FrameHessian>> frameHessians,
                                         std::shared_ptr<FrameHessian> fh_right, CalibHessian Hcalib)
{
    assert(frameHessians.size() > 0);
    lastRef = frameHessians.back();

    makeCoarseDepthL0(frameHessians, fh_right, Hcalib);

    refFrameID = lastRef->shell->id;
    lastRef_aff_g2l = lastRef->aff_g2l();

    firstCoarseRMSE = -1;

}
bool CoarseTracker::trackNewestCoarse(
    std::shared_ptr<FrameHessian> newFrameHessian,
    SE3& lastToNew_out, AffLight& aff_g2l_out,
    int coarsestLvl,
    Vec5 minResForAbort,
    Viewer::Output3D* wrap)
{
    debugPlot = setting_render_displayCoarseTrackingFull;
    debugPrint = false;

    assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

    lastResiduals.setConstant(NAN);
    lastFlowIndicators.setConstant(1000);


    newFrame = newFrameHessian;
    int maxIterations[] = {10, 20, 50, 50, 50};
    float lambdaExtrapolationLimit = 0.001f;

    SE3 refToNew_current = lastToNew_out;
    AffLight aff_g2l_current = aff_g2l_out;

    bool haveRepeated = false;


    for(int lvl = coarsestLvl; lvl >= 0; lvl--)
    {
        Mat88 H;
        Vec8 b;
        float levelCutoffRepeat = 1;
        Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current,
                              setting_coarseCutoffTH * levelCutoffRepeat);

        while(resOld[5] > 0.6 && levelCutoffRepeat < 50)
        {
            levelCutoffRepeat *= 2;
            resOld = calcRes(lvl, refToNew_current, aff_g2l_current,
                             setting_coarseCutoffTH * levelCutoffRepeat);

            if(!setting_debugout_runquiet)
            {
                LOG_INFO("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH * levelCutoffRepeat,
                         resOld[5]);
            }
        }

        calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

        float lambda = 0.01f;

        if(debugPrint)
        {
            Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure,
                                                       lastRef_aff_g2l, aff_g2l_current).cast<float>();
            LOG_INFO("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
                     lvl, -1, lambda, 1.0f,
                     "INITIA",
                     0.0f,
                     resOld[0] / resOld[1],
                     0, (int)resOld[1],
                     0.0f);
            std::cout << refToNew_current.log().transpose() << " AFF " << aff_g2l_current.vec().transpose() <<
                      " (rel " << relAff.transpose() << ")\n";
        }


        for(int iteration = 0; iteration < maxIterations[lvl]; iteration++)
        {
            Mat88 Hl = H;

            for(int i = 0; i < 8; i++)
            {
                Hl(i, i) *= (1 + lambda);
            }

            Vec8 inc = Hl.ldlt().solve(-b);

            if(setting_affineOptModeA < 0 && setting_affineOptModeB < 0)    // fix a, b
            {
                inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
                inc.tail<2>().setZero();
            }

            if(!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0) // fix b
            {
                inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
                inc.tail<1>().setZero();
            }

            if(setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0)) // fix a
            {
                Mat88 HlStitch = Hl;
                Vec8 bStitch = b;
                HlStitch.col(6) = HlStitch.col(7);
                HlStitch.row(6) = HlStitch.row(7);
                bStitch[6] = bStitch[7];
                Vec7 incStitch = HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
                inc.setZero();
                inc.head<6>() = incStitch.head<6>();
                inc[6] = 0;
                inc[7] = incStitch[6];
            }




            float extrapFac = 1;

            if(lambda < lambdaExtrapolationLimit)
            {
                extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
            }

            inc *= extrapFac;

            Vec8 incScaled = inc;
            incScaled.segment<3>(0) *= SCALE_XI_ROT;
            incScaled.segment<3>(3) *= SCALE_XI_TRANS;
            incScaled.segment<1>(6) *= SCALE_A;
            incScaled.segment<1>(7) *= SCALE_B;

            if(!std::isfinite(incScaled.sum()))
            {
                incScaled.setZero();
            }

            SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
            AffLight aff_g2l_new = aff_g2l_current;
            aff_g2l_new.a += incScaled[6];
            aff_g2l_new.b += incScaled[7];

            Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH * levelCutoffRepeat);

            bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

            if(debugPrint)
            {
                Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure,
                                                           lastRef_aff_g2l, aff_g2l_new).cast<float>();
                LOG_INFO("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
                         lvl, iteration, lambda,
                         extrapFac,
                         (accept ? "ACCEPT" : "REJECT"),
                         resOld[0] / resOld[1],
                         resNew[0] / resNew[1],
                         (int)resOld[1], (int)resNew[1],
                         inc.norm());
                std::cout << refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() << " (rel "
                          << relAff.transpose() << ")\n";
            }

            if(accept)
            {
                calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
                resOld = resNew;
                // printf("accepted with res: %f\n", resOld[0]/resOld[1]);
                aff_g2l_current = aff_g2l_new;
                refToNew_current = refToNew_new;
                lambda *= 0.5;
            }
            else
            {
                lambda *= 4;

                if(lambda < lambdaExtrapolationLimit)
                {
                    lambda = lambdaExtrapolationLimit;
                }
            }

            if(!(inc.norm() > 1e-3))
            {
                if(debugPrint)
                {
                    LOG_INFO("inc too small, break!\n");
                }

                break;
            }
        }

        // set last residual for that level, as well as flow indicators.
        lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
        lastFlowIndicators = resOld.segment<3>(2);

        if(lastResiduals[lvl] > 1.5 * minResForAbort[lvl])
        {
            return false;
        }


        if(levelCutoffRepeat > 1 && !haveRepeated)
        {
            lvl++;
            haveRepeated = true;
            LOG_INFO("REPEAT LEVEL!\n");
        }
    }

    // set!
    lastToNew_out = refToNew_current;
    aff_g2l_out = aff_g2l_current;


    if((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2f))
       || (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200.f)))
    {
        return false;
    }

    Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure,
                                               lastRef_aff_g2l, aff_g2l_out).cast<float>();

    if((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5))
       || (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
    {
        return false;
    }



    if(setting_affineOptModeA < 0)
    {
        aff_g2l_out.a = 0;
    }

    if(setting_affineOptModeB < 0)
    {
        aff_g2l_out.b = 0;
    }

    return true;
}

void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt,
                                       std::vector<std::shared_ptr<Viewer::Output3D>>& wraps)
{
    if(w[1] == 0)
    {
        return;
    }


    int lvl = 0;

    {
        std::vector<float> allID;

        for(int i = 0; i < h[lvl]*w[lvl]; i++)
        {
            if(idepth[lvl][i] > 0)
            {
                allID.push_back(idepth[lvl][i]);
            }
        }

        std::sort(allID.begin(), allID.end());
        int n = static_cast<int>(allID.size()) - 1;

        if (n < 0)
        {
            return;
        }

        float minID_new = allID[(int)(n * 0.05)];
        float maxID_new = allID[(int)(n * 0.95)];

        float minID, maxID;
        minID = minID_new;
        maxID = maxID_new;

        if(minID_pt != 0 && maxID_pt != 0)
        {
            if(*minID_pt < 0 || *maxID_pt < 0)
            {
                *maxID_pt = maxID;
                *minID_pt = minID;
            }
            else
            {

                // slowly adapt: change by maximum 10% of old span.
                float maxChange = 0.3f * (*maxID_pt - *minID_pt);

                if(minID < *minID_pt - maxChange)
                {
                    minID = *minID_pt - maxChange;
                }

                if(minID > *minID_pt + maxChange)
                {
                    minID = *minID_pt + maxChange;
                }


                if(maxID < *maxID_pt - maxChange)
                {
                    maxID = *maxID_pt - maxChange;
                }

                if(maxID > *maxID_pt + maxChange)
                {
                    maxID = *maxID_pt + maxChange;
                }

                *maxID_pt = maxID;
                *minID_pt = minID;
            }
        }

        cv::Mat mf(h[lvl], w[lvl], CV_8UC3, cv::Scalar::all(0));

        for(int i = 0; i < h[lvl]*w[lvl]; i++)
        {
            int c = static_cast<int>(lastRef->dIp[lvl].ptr<Eigen::Vector3f>()[i][0] * 0.9f);

            if(c > 255)
            {
                c = 255;
            }

            mf.at<cv::Vec3b>(i) = cv::Vec3b(c, c, c);
        }

        int wl = w[lvl];

        for(int y = 3; y < h[lvl] - 3; y++)
            for(int x = 3; x < wl - 3; x++)
            {
                int idx = x + y * wl;
                float sid = 0, nid = 0;
                float* bp = idepth[lvl] + idx;

                if(bp[0] > 0)
                {
                    sid += bp[0];
                    nid++;
                }

                if(bp[1] > 0)
                {
                    sid += bp[1];
                    nid++;
                }

                if(bp[-1] > 0)
                {
                    sid += bp[-1];
                    nid++;
                }

                if(bp[wl] > 0)
                {
                    sid += bp[wl];
                    nid++;
                }

                if(bp[-wl] > 0)
                {
                    sid += bp[-wl];
                    nid++;
                }

                if(bp[0] > 0 || nid >= 3)
                {
                    float id = ((sid / nid) - minID) / ((maxID - minID));
                    cv::circle(mf, cv::Point(x, y), 2, makeJet3B(id));
                }
            }

        //IOWrap::displayImage("coarseDepth LVL0", &mf, false);

        for(auto& ow : wraps)
        {
            ow->pushDepthImage(mf);
        }

        if(debugSaveImages)
        {
            char buf[1000];
            snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png", lastRef->shell->id, refFrameID);
            cv::imwrite(buf, mf);
        }

    }
}



void CoarseTracker::debugPlotIDepthMapFloat(std::vector<std::shared_ptr<Viewer::Output3D>>& wraps)
{
    if(w[1] == 0)
    {
        return;
    }

    int lvl = 0;
    cv::Mat mim(h[lvl], w[lvl], CV_32F, idepth[lvl], cv::Mat::AUTO_STEP);

    for(auto& ow : wraps)
    {
        ow->pushDepthImageFloat(mim);
    }
}











CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
{
    fwdWarpedIDDistFinal = new float[ww * hh / 4];

    bfsList1 = new Eigen::Vector2i[ww * hh / 4];
    bfsList2 = new Eigen::Vector2i[ww * hh / 4];

    int fac = 1 << (pyrLevelsUsed - 1);


    //coarseProjectionGrid = new PointFrameResidual*[2048 * (ww * hh / (fac * fac))];
    //coarseProjectionGridNum = new int[ww * hh / (fac * fac)];

    w[0] = h[0] = 0;
}
CoarseDistanceMap::~CoarseDistanceMap()
{
    delete[] fwdWarpedIDDistFinal;
    delete[] bfsList1;
    delete[] bfsList2;
    //delete[] coarseProjectionGrid;
    //delete[] coarseProjectionGridNum;
}





void CoarseDistanceMap::makeDistanceMap(
    std::vector<std::shared_ptr<FrameHessian>> frameHessians,
    std::shared_ptr<FrameHessian> frame)
{
    int w1 = w[1];
    int h1 = h[1];
    int wh1 = w1 * h1;

    for(int i = 0; i < wh1; i++)
    {
        fwdWarpedIDDistFinal[i] = 1000;
    }


    // make coarse tracking templates for latstRef.
    int numItems = 0;

    for(auto& fh : frameHessians)
    {
        if(frame.get() == fh.get())
        {
            continue;
        }

        SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
        Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
        Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

        for(PointHessian* ph : fh->pointHessians)
        {
            assert(ph->status == PointHessian::ACTIVE);
            Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * ph->idepth_scaled;
            int u = static_cast<float>(ptp[0] / ptp[2] + 0.5f);
            int v = static_cast<float>(ptp[1] / ptp[2] + 0.5f);

            if(!(u > 0 && v > 0 && u < w[1] && v < h[1]))
            {
                continue;
            }

            fwdWarpedIDDistFinal[u + w1 * v] = 0;
            bfsList1[numItems] = Eigen::Vector2i(u, v);
            numItems++;
        }
    }

    growDistBFS(numItems);
}



void CoarseDistanceMap::growDistBFS(int bfsNum)
{
    assert(w[0] != 0);
    int w1 = w[1], h1 = h[1];

    for(int k = 1; k < 40; k++)
    {
        int bfsNum2 = bfsNum;
        std::swap(bfsList1, bfsList2);
        bfsNum = 0;
        float kk = static_cast<float>(k);

        if(k % 2 == 0)
        {
            for(int i = 0; i < bfsNum2; i++)
            {
                int x = bfsList2[i][0];
                int y = bfsList2[i][1];

                if(x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1)
                {
                    continue;
                }

                int idx = x + y * w1;



                if(fwdWarpedIDDistFinal[idx + 1] > kk)
                {
                    fwdWarpedIDDistFinal[idx + 1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx - 1] > kk)
                {
                    fwdWarpedIDDistFinal[idx - 1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx + w1] > kk)
                {
                    fwdWarpedIDDistFinal[idx + w1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx - w1] > kk)
                {
                    fwdWarpedIDDistFinal[idx - w1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
                    bfsNum++;
                }
            }
        }
        else
        {
            for(int i = 0; i < bfsNum2; i++)
            {
                int x = bfsList2[i][0];
                int y = bfsList2[i][1];

                if(x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1)
                {
                    continue;
                }

                int idx = x + y * w1;

                if(fwdWarpedIDDistFinal[idx + 1] > kk)
                {
                    fwdWarpedIDDistFinal[idx + 1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx - 1] > kk)
                {
                    fwdWarpedIDDistFinal[idx - 1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx + w1] > kk)
                {
                    fwdWarpedIDDistFinal[idx + w1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx - w1] > kk)
                {
                    fwdWarpedIDDistFinal[idx - w1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx + 1 + w1] > kk)
                {
                    fwdWarpedIDDistFinal[idx + 1 + w1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y + 1);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx - 1 + w1] > kk)
                {
                    fwdWarpedIDDistFinal[idx - 1 + w1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y + 1);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx - 1 - w1] > kk)
                {
                    fwdWarpedIDDistFinal[idx - 1 - w1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y - 1);
                    bfsNum++;
                }

                if(fwdWarpedIDDistFinal[idx + 1 - w1] > kk)
                {
                    fwdWarpedIDDistFinal[idx + 1 - w1] = kk;
                    bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y - 1);
                    bfsNum++;
                }
            }
        }
    }
}


void CoarseDistanceMap::addIntoDistFinal(int u, int v)
{
    if(w[0] == 0)
    {
        return;
    }

    bfsList1[0] = Eigen::Vector2i(u, v);
    fwdWarpedIDDistFinal[u + w[1]*v] = 0;
    growDistBFS(1);
}



void CoarseDistanceMap::makeK(CalibHessian* HCalib)
{
    w[0] = wG[0];
    h[0] = hG[0];

    fx[0] = HCalib->fxl();
    fy[0] = HCalib->fyl();
    cx[0] = HCalib->cxl();
    cy[0] = HCalib->cyl();

    for (int level = 1; level < pyrLevelsUsed; ++ level)
    {
        w[level] = w[0] >> level;
        h[level] = h[0] >> level;
        fx[level] = fx[level - 1] * 0.5f;
        fy[level] = fy[level - 1] * 0.5f;
        cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5f;
        cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5f;
    }

    for (int level = 0; level < pyrLevelsUsed; ++ level)
    {
        K[level]  << fx[level], 0.0f, cx[level], 0.0f, fy[level], cy[level], 0.0f, 0.0f, 1.0f;
        Ki[level] = K[level].inverse();
        fxi[level] = Ki[level](0, 0);
        fyi[level] = Ki[level](1, 1);
        cxi[level] = Ki[level](0, 2);
        cyi[level] = Ki[level](1, 2);
    }
}

}
