#pragma once

#include <string>
#include "OpenCLHelper.hpp"

namespace dso
{
namespace OCLKernels
{

static const std::string KernelCalcRes = "__kernel void calcRes(\
        __global const float* restrict in_id,\
        __global const float* restrict in_x,\
        __global const float* restrict in_y,\
        const float16 RKi_t_fxfycxcy,\
        const float16 Ki_affL_cutoffTH_maxEnergy,\
        int4 kWidth_wl_hl_lvl,\
        __global const float* restrict lpc_color,\
        __global const float* restrict dINewl,\
        volatile __global uint* numTermsInWarped,\
        __global float* restrict buf_warped,\
        __global float* rs)\
        {                                       \
            unsigned int i = get_global_id(0);  \
            int kwidth = kWidth_wl_hl_lvl.s0;   \
            if(i >= kwidth)                     \
            {\
                return;\
            }\
            float id = in_id[i];                \
            float x = in_x[i];                  \
            float y = in_y[i];                  \
                                                \
            int lvl = kWidth_wl_hl_lvl.s3;      \
            int wl = kWidth_wl_hl_lvl.s1;       \
            int hl = kWidth_wl_hl_lvl.s2;       \
            float2 affLL = (float2)(Ki_affL_cutoffTH_maxEnergy.s3, Ki_affL_cutoffTH_maxEnergy.s7); \
            float setting_huberTH = Ki_affL_cutoffTH_maxEnergy.sE;                                \
            float maxEnergy = Ki_affL_cutoffTH_maxEnergy.sD;                                      \
            float cutoffTH = Ki_affL_cutoffTH_maxEnergy.sC;                                       \
                                                                                                   \
            float4 tid = id * (float4)(RKi_t_fxfycxcy.s3, RKi_t_fxfycxcy.s7, RKi_t_fxfycxcy.sB, 1.0f);               \
            float4 fxfycxcy = (float4)(RKi_t_fxfycxcy.sB, RKi_t_fxfycxcy.sC, RKi_t_fxfycxcy.sD, RKi_t_fxfycxcy.sE);\
                                                                                                              \
            float4 pt = (float4)(RKi_t_fxfycxcy.s0 * x + RKi_t_fxfycxcy.s1 * y + RKi_t_fxfycxcy.s2  + tid.s0, \
                                 RKi_t_fxfycxcy.s4 * x + RKi_t_fxfycxcy.s5 * y + RKi_t_fxfycxcy.s6  + tid.s1, \
                                 RKi_t_fxfycxcy.s8 * x + RKi_t_fxfycxcy.s9 * y + RKi_t_fxfycxcy.sA + tid.s2, \
                                 1.0f);  \
                                                      \
            float u = pt.s0 / pt.s2;                  \
            float v = pt.s1 / pt.s2;                  \
            float Ku = fxfycxcy.s0 * u + fxfycxcy.s2; \
            float Kv = fxfycxcy.s1 * v + fxfycxcy.s3; \
            float new_idepth = id / pt.s2;            \
                                                      \
            if (lvl == 0 && i % 32 == 0)              \
            {                                         \
                float4 ptT = (float4)(Ki_affL_cutoffTH_maxEnergy.s0 * x + Ki_affL_cutoffTH_maxEnergy.s1 * y + Ki_affL_cutoffTH_maxEnergy.s2  + tid.s0, \
                                      Ki_affL_cutoffTH_maxEnergy.s4 * x + Ki_affL_cutoffTH_maxEnergy.s5 * y + Ki_affL_cutoffTH_maxEnergy.s6  + tid.s1, \
                                      Ki_affL_cutoffTH_maxEnergy.s8 * x + Ki_affL_cutoffTH_maxEnergy.s9 * y + Ki_affL_cutoffTH_maxEnergy.sA + tid.s2, \
                                      1.0f);  \
                                                            \
                float uT = ptT.s0 / ptT.s2;                 \
                float vT = ptT.s1 / ptT.s2;                 \
                float KuT = fxfycxcy.s0 * uT + fxfycxcy.s2; \
                float KvT = fxfycxcy.s1 * vT + fxfycxcy.s3; \
                                                            \
                float4 ptT2 = (float4)(Ki_affL_cutoffTH_maxEnergy.s0 * x + Ki_affL_cutoffTH_maxEnergy.s1 * y + Ki_affL_cutoffTH_maxEnergy.s2  - tid.s0,\
                                       Ki_affL_cutoffTH_maxEnergy.s4 * x + Ki_affL_cutoffTH_maxEnergy.s5 * y + Ki_affL_cutoffTH_maxEnergy.s6  - tid.s1,\
                                       Ki_affL_cutoffTH_maxEnergy.s8 * x + Ki_affL_cutoffTH_maxEnergy.s9 * y + Ki_affL_cutoffTH_maxEnergy.sA - tid.s2,\
                                       1.0f); \
                 \
                float uT2 = ptT2.s0 / ptT2.s2;  \
                float vT2 = ptT2.s1 / ptT2.s2;  \
                float KuT2 = fxfycxcy.s0 * uT2 + fxfycxcy.s2; \
                float KvT2 = fxfycxcy.s1 * vT2 + fxfycxcy.s3; \
                 \
                float4 pt3 = (float4)(RKi_t_fxfycxcy.s0 * x + RKi_t_fxfycxcy.s1 * y + RKi_t_fxfycxcy.s2  - tid.s0, \
                                      RKi_t_fxfycxcy.s4 * x + RKi_t_fxfycxcy.s5 * y + RKi_t_fxfycxcy.s6  - tid.s1, \
                                      RKi_t_fxfycxcy.s8 * x + RKi_t_fxfycxcy.s9 * y + RKi_t_fxfycxcy.sA - tid.s2, \
                                      1.0f);                \
                                                            \
                float u3 = pt3.s0 / pt3.s2;                 \
                float v3 = pt3.s1 / pt3.s2;                 \
                float Ku3 = fxfycxcy.s0 * u3 + fxfycxcy.s2; \
                float Kv3 = fxfycxcy.s1 * v3 + fxfycxcy.s3; \
                                                            \
                rs[6 * i + 2] = (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);    \
                rs[6 * i + 2] = (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);\
                rs[6 * i + 4] = (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);        \
                rs[6 * i + 4] = (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);    \
                rs[6 * i + 3] = 2;                                                \
            }                                                                 \
                                                                              \
            if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0))\
            {\
                return;\
            }\
                                            \
            float refColor = lpc_color[i];  \
            float4 hitColor;                 \
                                       \
            int ix = (int)Ku;           \
            int iy = (int)Kv;           \
            float dx = Ku - ix;         \
            float dy = Kv - iy;         \
            float dxdy = dx * dy;       \
            __global const float* bp = dINewl + 3 * (ix + iy * wl);     \
                                                               \
            hitColor = (float4) (dxdy * (*(bp + 3 * (1 + wl)))          \
                     + (dy - dxdy) * (*(bp + 3 * wl))          \
                     + (dx - dxdy) * (*(bp + 3))               \
                     + (1 - dx - dy + dxdy) * (*bp),           \
                     dxdy* (*(bp + 3 * (1 + wl) + 1))          \
                     + (dy - dxdy) * (*(bp + 3 * wl + 1))      \
                     + (dx - dxdy) * (*(bp + 3 + 1))           \
                     + (1 - dx - dy + dxdy) * (*(bp + 1)),     \
                     dxdy* (*(bp + 3 * (1 + wl) + 2))          \
                     + (dy - dxdy) * (*(bp + 3 * wl + 2))      \
                     + (dx - dxdy) * (*(bp + 3 + 2))           \
                     + (1 - dx - dy + dxdy) * (*(bp + 2)), 1.0f);     \
                                                              \
                                                               \
            if (isfinite(hitColor.s0) == 0)\
            {\
                return;\
            }\
                                                                   \
            float residual = hitColor.s0 - (float)(affLL.s0 * refColor + affLL.s1); \
            float hw;   \
            if(fabs(residual) < setting_huberTH)\
            {\
                hw = 1;  \
            } \
            else \
            {\
                hw = setting_huberTH / fabs(residual); \
            }\
                                                                  \
                                                                  \
            if (fabs(residual) > cutoffTH)                        \
            {                                                     \
                rs[6 * i + 0] = maxEnergy;                            \
                rs[6 * i + 1] = 1;                                       \
                rs[6 * i + 5] = 1;                                       \
            }\
            else                                                  \
            {                                                     \
                rs[6 * i + 0] = hw * residual * residual * (2 - hw);  \
                rs[6 * i + 1] = 1;                                       \
                                                                  \
                int listIndex = atomic_inc(numTermsInWarped);     \
                int li8 = 8 * listIndex;                          \
                buf_warped[li8 + 0] = new_idepth;               \
                buf_warped[li8 + 1] = u;                             \
                buf_warped[li8 + 2] = v;                             \
                buf_warped[li8 + 3] = hitColor.s1;                  \
                buf_warped[li8 + 4] = hitColor.s2;                  \
                buf_warped[li8 + 5] = residual;               \
                buf_warped[li8 + 6] = hw;                       \
                buf_warped[li8 + 7] = lpc_color[i];           \
            }                                                     \
        }";

static const std::string KernelHessian = "__kernel void hessian(			    \
        __global const float* restrict input,					                \
        const int kwidth,											            \
        const int kheight,											            \
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
            float dx = 0.0f;                                                                                                            \
            float dy = 0.0f;                                                                                                            \
                                                                                                                                       \
            /* Only inside the borders of the image (go from v = 1 to v = height - 2) */                                               \
            if (v_out > 0 && v_out < kheight - 1)                                                                                      \
            {                                                                                                                          \
                dx = 0.5f * (input[u_out + 1 + v_out * kwidth] - input[u_out - 1 + v_out * kwidth]);                                    \
                dy = 0.5f * (input[u_out + (v_out + 1) * kwidth] - input[u_out + (v_out - 1) * kwidth]);                                \
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
        __global       float* restrict outf,                                                        \
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

}
}
