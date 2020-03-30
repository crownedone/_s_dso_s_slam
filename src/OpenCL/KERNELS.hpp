#pragma once

#include <string>


static const std::string KernelHessian = "__kernel void hessian(			    \
        __global const float* restrict input,					                \
        const int kwidth,											            \
        __global       float* restrict out3f,                                   \
        __global       float* restrict outAbsGrad)							    \
        {																	    \
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
