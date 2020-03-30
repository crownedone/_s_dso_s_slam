#include "catch.hpp"
#include "../src/Logging.hpp"
#include "../src/IOWrapper/Input.hpp"
#include "../src/OpenCL/OpenCLHelper.hpp"
#include <StopWatch.hpp>
#include <Logging.hpp>

#include <opencv2/core/ocl.hpp>
//#include <opencv2/core/hal/interface.h>

// This is the whole hessian calculation part:
void hessianKernelCPU(int U, int V, const float* input, int kwidth, int kheight, float* out3f,
                      float* outAbsGrad)
{
    for (int v = 0; v < V; ++v)
    {
        for(int u = 0; u < U; ++u)
        {
            int u_out = u; // get_global_id(0);
            int v_out = v; // get_global_id(1);

            // OpenCV Specific: Kernels may be run with slightly bigger vSize according to alignment. This prevents unwanted access
            if (u_out >= kwidth)
            {
                continue; // return;
            }

            float color = input[u_out + v_out * kwidth];

            // Hesse Calculation:
            float dx = 0.f;
            float dy = 0.f;

            // Only inside the borders of the image (go from v = 1 to v = height - 2)
            if (v_out > 0 && v_out < kheight - 1)
            {
                dx = .5f * (input[u_out + 1 + v_out * kwidth] - input[u_out - 1 + v_out * kwidth]);
                dy = .5f * (input[u_out + (v_out + 1) * kwidth] - input[u_out + (v_out - 1) * kwidth]);
            }

            // Save to out3f
            out3f[3 * (u_out + v_out * kwidth) + 0] = color;
            out3f[3 * (u_out + v_out * kwidth) + 1] = dx;
            out3f[3 * (u_out + v_out * kwidth) + 2] = dy;

            // Save to absGrad
            outAbsGrad[u_out + v_out * kwidth] = dx * dx + dy * dy;
        };
    }

}

void bGradKernelCPU(int U, int V, const float* input3f, const float* inputAbsGrad, int kwidth,
                    const float* B,
                    float* outAbsGrad)
{
    for (int v = 0; v < V; ++v)
    {
        for (int u = 0; u < U; ++u)
        {
            int u_out = u; // get_global_id(0);
            int v_out = v; // get_global_id(1);

            // OpenCV Specific: Kernels may be run with slightly bigger vSize according to alignment. This prevents unwanted access
            if (u_out >= kwidth)
            {
                continue; // return;
            }

            float color = input3f[3 * (u_out + v_out * kwidth)];

            int c = (int)(color + 0.5f);

            // GPU:
            // c = clamp(c, 5, 250);
            if (c < 5)
            {
                c = 5;
            }

            if (c > 250)
            {
                c = 250;
            }

            float gw = B[c + 1] - B[c];

            outAbsGrad[u_out + v_out * kwidth] = inputAbsGrad[u_out + v_out * kwidth] * gw * gw;
        }
    }
};

void pyrDownKernelCPU(int U, int V, const float* inputf, int kwidth, float* outf, int inputWidth)
{
    for (int v = 0; v < V; ++v)
    {
        for (int u = 0; u < U; ++u)
        {
            int u_out = u; // get_global_id(0);
            int v_out = v; // get_global_id(1);

            // OpenCV Specific: Kernels may be run with slightly bigger vSize according to alignment. This prevents unwanted access
            if (u_out >= kwidth)
            {
                continue; // return;
            }

            int u_in0 = u_out * 2;
            int u_in1 = u_out * 2 + 1;
            int v_in0 = v_out * 2;
            int v_in1 = v_out * 2 + 1;
            /* Pyr down */
            outf[u_out + v_out * kwidth] = 0.25f * (inputf[u_in0 + v_in0 * inputWidth] +
                                                    inputf[u_in1 + v_in0 * inputWidth] +
                                                    inputf[u_in0 + v_in1 * inputWidth] +
                                                    inputf[u_in1 + v_in1 * inputWidth]);

        }
    }
};

static const std::string KernelHessian = "__kernel void hessian(			\
        __global const float* restrict input,					        \
        const int kwidth,											        \
        __global       float* restrict out3f,                            \
        __global       float* restrict outAbsGrad)							\
        {																	\
            int u_out = get_global_id(0);                                       \
            int v_out = get_global_id(1);                                       \
                                                                                \
            /* OpenCV Specific: Kernels may be run with slightly bigger vSize according to alignment. This prevents unwanted access */ \
            if (u_out >= kwidth)                                                                                                       \
            {                                                                                                                          \
                return;                                                                                                                \
            }                                                                                                                          \
                                                                                                                                       \
            float color = input[u_out + v_out * kwidth];                                                                               \
                                                                                                                                       \
            /* Hesse Calculation: */                                                                                                   \
            float dx = 0.f;                                                                                                            \
            float dy = 0.f;                                                                                                            \
                                                                                                                                       \
            /* Only inside the borders of the image (go from v = 1 to v = height - 2) */                                               \
            if (v_out > 0 && v_out < kheight - 1)                                                                                      \
            {                                                                                                                          \
                dx = .5f * (input[u_out + 1 + v_out * kwidth] - input[u_out - 1 + v_out * kwidth]);                                    \
                dy = .5f * (input[u_out + (v_out + 1) * kwidth] - input[u_out + (v_out - 1) * kwidth]);                                \
            }                                                                                                                          \
                                                                                                                                       \
            out3f[3 * (u_out + v_out * kwidth) + 0] = color;                                                                           \
            out3f[3 * (u_out + v_out * kwidth) + 1] = dx;                                                                              \
            out3f[3 * (u_out + v_out * kwidth) + 2] = dy;                                                                              \
                                                                                                                                       \
            outAbsGrad[u_out + v_out * kwidth] = dx * dx + dy * dy;                                                                    \
        }";

static const std::string KernelBGrad = "__kernel void bGrad(			                            \
        __global const float* restrict input3f,					                                    \
        __global       float*          inOutAbsGrad,					                            \
        const int kwidth,											                                \
        __global const float* restrict B)                                                           \
        {                                                                                           \
            int u_out = get_global_id(0);                                                           \
            int v_out = get_global_id(1);                                                           \
                                                                                                    \
            if (u_out >= kwidth)                                                                    \
            {                                                                                       \
                return;                                                                             \
            }                                                                                       \
                                                                                                    \
            float color = input3f[3 * (u_out + v_out * kwidth)];                                    \
                                                                                                    \
            int c = (int)(color + 0.5f);                                                            \
            c = clamp(c, 5, 250);                                                                   \
            float gw = B[c + 1] - B[c];                                                             \
                                                                                                    \
            inOutAbsGrad[u_out + v_out * kwidth] *= gw * gw;                                        \
        }";

static const std::string KernelPyrDown = "__kernel void pyrDown(			                        \
        __global const float* restrict inputf,					                                    \
        const int kwidth,											                                \
        __global const float* restrict outf,                                                        \
        int inputWidth)                                                                             \
        {                                                                                           \
            int u_out = get_global_id(0);                                                           \
            int v_out = get_global_id(1);                                                           \
                                                                                                    \
            if (u_out >= kwidth)                                                                    \
            {                                                                                       \
                return;                                                                             \
            }                                                                                       \
                                                                                                    \
            int u_in0 = u_out * 2;                                                                  \
            int u_in1 = u_out * 2 + 1;                                                              \
            int v_in0 = v_out * 2;                                                                  \
            int v_in1 = v_out * 2 + 1;                                                              \
            /* Pyr down */                                                                          \
            outf[u_out + v_out * kwidth] = 0.25f * (inputf[u_in0 + v_in0 * inputWidth] +            \
                                                    inputf[u_in1 + v_in0 * inputWidth] +            \
                                                    inputf[u_in0 + v_in1 * inputWidth] +            \
                                                    inputf[u_in1 + v_in1 * inputWidth]);            \
        }";


void makeImagesGPU(cv::Mat color, std::vector<cv::Mat>& dIp, std::vector<cv::Mat>& absSquaredGrad,
                   int pyrLevelsUsed,
                   float* BGrad = nullptr)
{
    assert(color.empty());

    // Kernels:
    cv::String hash1, hash2, hash3;
    // Maybe do as members?
    cv::ocl::ProgramSource bGradKernelOCV("makeImages", "bGrad", KernelBGrad, "");
    cv::ocl::ProgramSource hessianKernelOCV("makeImages", "hessian", KernelHessian, "");
    cv::ocl::ProgramSource pyrDownKernelOCV("makeImages", "pyrDown", KernelPyrDown, "");

    // Initialize on GPU:
    std::vector<cv::UMat> colorUMat;
    std::vector<cv::UMat> dIpUMat;
    std::vector<cv::UMat> absSquaredGradUMat;

    //
    cv::Mat B;
    cv::UMat BUMat;

    if (BGrad)
    {
        // Wrap Mat around the memory of BGrad (only for upload)
        B = cv::Mat(1, 255, CV_32FC1, BGrad, cv::Mat::AUTO_STEP);
        // Upload
        B.copyTo(BUMat);
    }

    for (int i = 0; i < pyrLevelsUsed; i++)
    {
        int w = color.cols / std::pow(2, i);
        int h = color.rows / std::pow(2, i);
        colorUMat.push_back(cv::UMat(h, w, CV_32FC1, cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY));
        dIpUMat.push_back(cv::UMat(h, w, CV_32FC3, cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY));
        absSquaredGradUMat.push_back(cv::UMat(h, w, CV_32FC1,
                                              cv::UMatUsageFlags::USAGE_ALLOCATE_DEVICE_MEMORY));
    }

    // Upload the color image:
    color.copyTo(colorUMat[0]);

    for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
    {
        // Run pyr Down kernel
        if (lvl > 0)
        {
            int kwidth = colorUMat[lvl].cols;
            int kheight = colorUMat[lvl].rows;
            int inputWidth = colorUMat[lvl - 1].cols;
            REQUIRE(RunKernel("pyrDown", pyrDownKernelOCV,
            {
                cv::ocl::KernelArg::PtrReadOnly(colorUMat[lvl - 1]),
                cv::ocl::KernelArg::Constant(&kwidth, sizeof(int)),
                cv::ocl::KernelArg::PtrWriteOnly(colorUMat[lvl]),
                cv::ocl::KernelArg::Constant(&inputWidth, sizeof(int))
            },
            { (size_t)(kwidth), (size_t)(kheight), (size_t)(0) },
            2, // Only use 2 dimensions (width, height)
            false));
        }

        int kwidth = colorUMat[lvl].cols;
        int kheight = colorUMat[lvl].rows;
        REQUIRE(RunKernel("hessian", hessianKernelOCV,
        {
            cv::ocl::KernelArg::PtrReadOnly(colorUMat[lvl]),
            cv::ocl::KernelArg::Constant(&kwidth, sizeof(int)),
            cv::ocl::KernelArg::PtrWriteOnly(dIpUMat[lvl]),
            cv::ocl::KernelArg::PtrWriteOnly(absSquaredGradUMat[lvl])
        },
        {(size_t)(kwidth), (size_t)(kheight), (size_t)(0)},
        2, // Only use 2 dimensions (width, height)
        false));

        // Only if needed (if input has BGrad):
        if (BGrad)
        {
            REQUIRE(RunKernel("bGrad", bGradKernelOCV,
            {
                cv::ocl::KernelArg::PtrReadOnly(dIpUMat[lvl]),              // input3f
                cv::ocl::KernelArg::PtrReadWrite(absSquaredGradUMat[lvl]),   // inputAbsGrad
                cv::ocl::KernelArg::Constant(&kwidth, sizeof(int)),         // kwidth
                cv::ocl::KernelArg::PtrReadOnly(BUMat)
            },
            { (size_t)(kwidth), (size_t)(kheight), (size_t)(0) },
            2, // Only use 2 dimensions (width, height)
            false));
        }
    }


    // Download Everything:
    dIp.resize(dIpUMat.size());                          // Create empy placeholder
    absSquaredGrad.resize(absSquaredGradUMat.size());    // Create empy placeholder

    for (int lvl = 0; lvl < pyrLevelsUsed; ++lvl)
    {
        // CopyTo 'dowloads' the Mat from GPU memory to the desired location.
        dIpUMat[lvl].copyTo(dIp[lvl]);
        absSquaredGradUMat[lvl].copyTo(absSquaredGrad[lvl]);
    }
}





TEST_CASE("OpenCL Test", "[OpenCL][MakeImages]")
{


}

