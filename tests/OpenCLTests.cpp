#include "catch.hpp"
#include "../src/Logging.hpp"
#include "../src/IOWrapper/Input.hpp"

#include <StopWatch.hpp>
#include <Logging.hpp>

#include <opencv2/core/ocl.hpp>
//#include <opencv2/core/hal/interface.h>


void binning(const cv::Mat& inputImage, cv::Mat& outputImage)
{

    outputImage.create(inputImage.size() / 2, CV_32F);

    for (int v = 0; v < outputImage.rows; ++v)
        for (int u = 0; u < outputImage.cols; ++u)
        {
            outputImage.at<float>(v, u) = ((int)(inputImage.at<uchar>(v * 2 + 0, u * 2 + 0)) +
                                           (int)(inputImage.at<uchar>(v * 2 + 0, u * 2 + 1)) +
                                           (int)(inputImage.at<uchar>(v * 2 + 1, u * 2 + 0)) +
                                           (int)(inputImage.at<uchar>(v * 2 + 1, u * 2 + 1))) / 4.f;
        }
}

TEST_CASE("OpenCL Test", "[OpenCL][SimpleKernel]")
{

    std::string kernelBinning =
        "__kernel void Binning(										\
        __global const uchar* restrict inputImage,					\
        const int kwidth,											\
        __global       float* restrict outputImage,					\
        const int inputWidth)										\
        {																	\
        uint u_out = get_global_id(0);										\
        uint v_out = get_global_id(1);										\
        uint outputWidth = kwidth;											\
                                                                            \
        if (u_out >= kwidth)                                                \
        {                                                                   \
            return;                                                         \
        }                                                                   \
                                                                            \
        int sum = 0;                                                        \
        for (uint dv = 0; dv < 2; ++dv)                                     \
        {                                                                   \
            for (uint du = 0; du < 2; ++du)                                 \
            {                                                               \
                uint u_in = u_out * 2 + du;                                 \
                uint v_in = v_out * 2 + dv;                                 \
                                                                            \
                sum += (int)(inputImage[u_in + v_in * inputWidth]);         \
                                                                            \
            }                                                               \
        }                                                                   \
                                                                            \
        float out = (float)(sum / (2 * 2));                                 \
        outputImage[u_out + v_out * outputWidth] = out;                     \
        }";

    cv::String hash;
    cv::ocl::ProgramSource binningKernelSource("general", "Binning", kernelBinning, hash);

    // Generate input, etc

    cv::Mat input(768, 1024, CV_8U);
    input.forEach<uchar>([](uchar & pixel, const int* position)
    {
        pixel = (std::rand() % 255);
    });


    StopWatch sw;


    cv::ocl::Queue q = cv::ocl::Queue::getDefault();

    for (int i = 0; i < 100; ++i)
    {
        cv::Mat output(384, 512, CV_32F);

        cv::ocl::Kernel k("Binning", binningKernelSource, "");

        REQUIRE(!k.empty());

        cv::UMat src1 = input.getUMat(cv::ACCESS_READ);
        cv::UMat dst = output.getUMat(cv::ACCESS_WRITE);

        int ow = input.cols;
        int kw = output.cols;
        cv::ocl::KernelArg inputImage = cv::ocl::KernelArg::PtrReadOnly(src1);
        cv::ocl::KernelArg kWidth = cv::ocl::KernelArg::Constant(&kw, sizeof(int));
        cv::ocl::KernelArg oWidth = cv::ocl::KernelArg::Constant(&ow, sizeof(int));
        cv::ocl::KernelArg outputImage = cv::ocl::KernelArg::PtrWriteOnly(dst);

        k.args(inputImage, kWidth, outputImage, oWidth);

        size_t globalsize[2] = { (size_t)output.cols, (size_t)output.rows };
        bool sync = true;
        k.run(2, globalsize, NULL, sync, q);

        LOG_INFO("Kernel run: %f ms", sw.restart());
    }


}

