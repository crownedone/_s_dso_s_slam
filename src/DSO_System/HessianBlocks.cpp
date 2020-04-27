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



#include "DSO_system/HessianBlocks.hpp"
#include "util/FrameShell.hpp"
#include "DSO_system/ImmaturePoint.hpp"
#include "OptimizationBackend/EnergyFunctionalStructs.hpp"
#include <Eigen/LU>
#include <opencv2/imgproc.hpp>
#include "StopWatch.hpp"
#include <OpenCL/KERNELS.hpp>

namespace dso
{


PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
{
    instanceCounter++;
    host = rawPoint->host;
    hasDepthPrior = false;

    idepth_hessian = 0;
    maxRelBaseline = 0;
    numGoodResiduals = 0;

    // set static values & initialization.
    u = rawPoint->u;
    v = rawPoint->v;
    assert(std::isfinite(rawPoint->idepth_max));
    //idepth_init = rawPoint->idepth_GT;

    my_type = rawPoint->my_type;

    setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) * 0.5);
    setPointStatus(PointHessian::INACTIVE);

    int n = patternNum;
    memcpy(color, rawPoint->color, sizeof(float)*n);
    memcpy(weights, rawPoint->weights, sizeof(float)*n);
    energyTH = rawPoint->energyTH;

    efPoint = 0;


}


void PointHessian::release()
{
    for(unsigned int i = 0; i < residuals.size(); i++)
    {
        delete residuals[i];
    }

    residuals.clear();
}


void FrameHessian::setStateZero(const Vec10& state_zero)
{
    assert(state_zero.head<6>().squaredNorm() < 1e-20);

    this->state_zero = state_zero;


    for(int i = 0; i < 6; i++)
    {
        Vec6 eps;
        eps.setZero();
        eps[i] = 1e-3;
        SE3 EepsP = Sophus::SE3<double>::exp(eps);
        SE3 EepsM = Sophus::SE3<double>::exp(-eps);
        SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
        SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
        nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
    }

    //nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
    //nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

    // scale change
    SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
    w2c_leftEps_P_x0.translation() *= 1.00001;
    w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
    SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
    w2c_leftEps_M_x0.translation() /= 1.00001;
    w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
    nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);


    nullspaces_affine.setZero();
    nullspaces_affine.topLeftCorner<2, 1>()  = Vec2(1, 0);
    assert(ab_exposure > 0);
    nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
};



void FrameHessian::release()
{
    // DELETE POINT
    // DELETE RESIDUAL
    for(unsigned int i = 0; i < pointHessians.size(); i++)
    {
        delete pointHessians[i];
    }

    for(unsigned int i = 0; i < pointHessiansMarginalized.size(); i++)
    {
        delete pointHessiansMarginalized[i];
    }

    for(unsigned int i = 0; i < pointHessiansOut.size(); i++)
    {
        delete pointHessiansOut[i];
    }

    for(unsigned int i = 0; i < immaturePoints.size(); i++)
    {
        delete immaturePoints[i];
    }


    pointHessians.clear();
    pointHessiansMarginalized.clear();
    pointHessiansOut.clear();
    immaturePoints.clear();
}

cv::ocl::ProgramSource KERNEL_makeImages_bGrad;
cv::ocl::ProgramSource KERNEL_makeImages_hessian;
cv::ocl::ProgramSource KERNEL_makeImages_pyrDown;
cv::Mat B;
cv::UMat BUMat;

void FrameHessian::makeImages(cv::UMat color, CalibHessian* HCalib)
{
    assert(setting_UseOpenCL);

// First time create kernels
    if (!KERNEL_makeImages_hessian.getImpl())
    {
        KERNEL_makeImages_bGrad = cv::ocl::ProgramSource("makeImages", "bGrad", OCLKernels::KernelBGrad, "");
        KERNEL_makeImages_hessian = cv::ocl::ProgramSource("makeImages", "hessian", OCLKernels::KernelHessian, "");
        KERNEL_makeImages_pyrDown = cv::ocl::ProgramSource("makeImages", "pyrDown", OCLKernels::KernelPyrDown, "");

        assert(KERNEL_makeImages_hessian.getImpl());
        assert(KERNEL_makeImages_bGrad.getImpl());
        assert(KERNEL_makeImages_pyrDown.getImpl());
    }

    //
    if (BUMat.empty() && setting_gammaWeightsPixelSelect == 1 && HCalib != 0)
    {
        float* BB = HCalib->getB();
        // Wrap Mat around the memory of BGrad (only for upload)
        B = cv::Mat(1, 256, CV_32FC1, BB, cv::Mat::AUTO_STEP);
        // Upload
        B.copyTo(BUMat);
    }

    for (int i = 0; i < pyrLevelsUsed; i++)
    {
        int w = color.cols / std::pow(2, i);
        int h = color.rows / std::pow(2, i);
        colorUMat[i] = cv::UMat(cv::UMat(h, w, CV_32FC1, cv::UMatUsageFlags::USAGE_DEFAULT));
        dIpUMat[i] = cv::UMat(cv::UMat(h, w, CV_32FC3, cv::UMatUsageFlags::USAGE_DEFAULT));
        absSquaredGradUMat[i] = cv::UMat(cv::UMat(h, w, CV_32FC1, cv::UMatUsageFlags::USAGE_DEFAULT));
    }

    // Upload the color image:
    colorUMat[0] = color;

    bool ksuccess = true;

    for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
    {
        // Run pyr Down kernel
        if (lvl > 0)
        {
            int kwidth = colorUMat[lvl].cols;
            int kheight = colorUMat[lvl].rows;
            int inputWidth = colorUMat[lvl - 1].cols;

            ksuccess &= ocl::RunKernel("pyrDown", KERNEL_makeImages_pyrDown,
            {
                cv::ocl::KernelArg::PtrReadOnly(colorUMat[lvl - 1]),
                cv::ocl::KernelArg::Constant(&kwidth, sizeof(int)),
                cv::ocl::KernelArg::PtrWriteOnly(colorUMat[lvl]),
                cv::ocl::KernelArg::Constant(&inputWidth, sizeof(int))
            },
            { (size_t)(kwidth), (size_t)(kheight), (size_t)(0) },
            2, // Only use 2 dimensions (width, height)
            false);

            if (!ksuccess)
            {
                LOG_ERROR("Kernel makeImages:pyrDown run error");
                exit(1);
            }

        }

        int kwidth = colorUMat[lvl].cols;
        int kheight = colorUMat[lvl].rows;
        ksuccess &= ocl::RunKernel("hessian", KERNEL_makeImages_hessian,
        {
            cv::ocl::KernelArg::PtrReadOnly(colorUMat[lvl]),
            cv::ocl::KernelArg::Constant(&kwidth, sizeof(int)),
            cv::ocl::KernelArg::Constant(&kheight, sizeof(int)),
            cv::ocl::KernelArg::PtrWriteOnly(dIpUMat[lvl]),
            cv::ocl::KernelArg::PtrWriteOnly(absSquaredGradUMat[lvl])
        },
        { (size_t)(kwidth), (size_t)(kheight), (size_t)(0) },
        2, // Only use 2 dimensions (width, height)
        false);

        if (!ksuccess)
        {
            LOG_ERROR("Kernel makeImages:hessian run error");
            exit(1);
        }

        // Only if needed (if input has BGrad):
        if (!BUMat.empty() && setting_gammaWeightsPixelSelect == 1 && HCalib != 0)
        {
            ksuccess &= ocl::RunKernel("bGrad", KERNEL_makeImages_bGrad,
            {
                cv::ocl::KernelArg::PtrReadOnly(dIpUMat[lvl]),              // input3f
                cv::ocl::KernelArg::PtrReadWrite(absSquaredGradUMat[lvl]),   // inputAbsGrad
                cv::ocl::KernelArg::Constant(&kwidth, sizeof(int)),         // kwidth
                cv::ocl::KernelArg::PtrReadOnly(BUMat)
            },
            { (size_t)(kwidth), (size_t)(kheight), (size_t)(0) },
            2, // Only use 2 dimensions (width, height)
            false);

            if (!ksuccess)
            {
                LOG_ERROR("Kernel makeImages:bGrad run error");
                exit(1);
            }
        }
    }

    // Download Everything:
    for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
    {
        // CopyTo 'dowloads' the Mat from GPU memory to the desired location.
        colorUMat[lvl].copyTo(colorMat[lvl]);
        dIpUMat[lvl].copyTo(dIp[lvl]);
        absSquaredGradUMat[lvl].copyTo(absSquaredGrad[lvl]);
    }

    // some pointer stuff needs setup here:
    dI_ptr = dIp[0].ptr<Eigen::Vector3f>();

}

void FrameHessian::makeImages(cv::Mat color, CalibHessian* HCalib)
{
    colorMat[0] = color;
    cv::Mat zero(hG[0], wG[0], CV_32FC1, cv::Scalar(0.f));
    std::vector<cv::Mat> arr = { color, zero, zero };
    cv::merge(arr, dIp[0]);
    absSquaredGrad[0] = cv::Mat(hG[0], wG[0], CV_32FC1);

    for(int i = 1; i < pyrLevelsUsed; i++)
    {
        colorMat[i] = cv::Mat(hG[i], wG[i], CV_32F);
        dIp[i] = cv::Mat(hG[i], wG[i], CV_32FC3);
        absSquaredGrad[i] = cv::Mat(hG[i], wG[i], CV_32FC1);
    }

    dI_ptr = dIp[0].ptr<Eigen::Vector3f>();

    // make d0
    int w = wG[0];
    int h = hG[0];

    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        if (lvl > 0)
        {
            cv::resize(colorMat[lvl - 1], colorMat[lvl], colorMat[lvl].size());
        }

        int wl = wG[lvl], hl = hG[lvl];
        float* col = colorMat[lvl].ptr<float>();
        Eigen::Vector3f* dI_l = dIp[lvl].ptr<Eigen::Vector3f>();
        float* dabs_l = absSquaredGrad[lvl].ptr<float>();

        for (int idx = 0; idx < wl * hl; idx++)
        {
            if (idx < wl || idx >= (wl * (hl - 1)))
            {
                dI_l[idx][0] = col[idx];
            }
            else
            {

                float dx = 0.5f * (col[idx + 1] - col[idx - 1]);
                float dy = 0.5f * (col[idx + wl] - col[idx - wl]);


                if (!std::isfinite(dx))
                {
                    dx = 0;
                }

                if (!std::isfinite(dy))
                {
                    dy = 0;
                }

                dI_l[idx][0] = col[idx];
                dI_l[idx][1] = dx;
                dI_l[idx][2] = dy;


                dabs_l[idx] = dx * dx + dy * dy;

                if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0)
                {
                    float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
                    dabs_l[idx] *= gw * gw; // convert to gradient of original color space (before removing response).
                }
            }
        }
    }
}

void FrameFramePrecalc::set(std::shared_ptr<FrameHessian> host, std::shared_ptr<FrameHessian> target, CalibHessian* HCalib )
{
    this->host = host;
    this->target = target;

    SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
    PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
    PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();



    SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
    PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
    PRE_tTll = (leftToLeft.translation()).cast<float>();
    distanceLL = leftToLeft.translation().norm();


    Mat33f K = Mat33f::Zero();
    K(0, 0) = HCalib->fxl();
    K(1, 1) = HCalib->fyl();
    K(0, 2) = HCalib->cxl();
    K(1, 2) = HCalib->cyl();
    K(2, 2) = 1;
    PRE_KRKiTll = K * PRE_RTll * K.inverse();
    PRE_RKiTll = PRE_RTll * K.inverse();
    PRE_KtTll = K * PRE_tTll;


    PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(),
                                               target->aff_g2l()).cast<float>();
    PRE_b0_mode = host->aff_g2l_0().b;
}

}

