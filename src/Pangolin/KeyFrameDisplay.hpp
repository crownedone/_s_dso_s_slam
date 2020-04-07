#pragma once

#undef Success
#include <Eigen/Core>

#include <pangolin/pangolin.h>

#include "IOWrapper/Output3D.hpp"

namespace Viewer
{

struct MyVertex
{
    float point[3];
    unsigned char color[4];
};

// stores a pointcloud associated to a Keyframe.
class KeyFrameDisplay
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    KeyFrameDisplay();
    ~KeyFrameDisplay();

    // copies points from KF over to internal buffer,
    // keeping some additional information so we can render it differently.
    void setFromKF(const KeyFrameView& kf);

    // copies points from KF over to internal buffer,
    // keeping some additional information so we can render it differently.
    void setFromF(const KeyFrameView& kf);

    // copies & filters internal data to GL buffer for rendering. if nothing to do: does nothing.
    bool refreshPC(bool canRefresh, float scaledTH, float absTH, int mode, float minBS, int sparsity);

    // renders cam & pointcloud.
    void drawCam(float lineWidth = 1, float* color = 0, float sizeFactor = 1);
    void drawPC(float pointSize);

    int id;
    bool active;
    Sophus::SE3d camToWorld;

    inline bool operator < (const KeyFrameDisplay& other) const
    {
        return (id < other.id);
    }


private:
    float fx, fy, cx, cy;
    float fxi, fyi, cxi, cyi;
    int width, height;

    float my_scaledTH, my_absTH, my_scale;
    int my_sparsifyFactor;
    int my_displayMode;
    float my_minRelBS;
    bool needRefresh;


    size_t numSparsePoints;
    int numSparseBufferSize;
    InputPointSparse<8>* originalInputSparse;


    bool bufferValid;
    int numGLBufferPoints;
    int numGLBufferGoodPoints;
    pangolin::GlBuffer vertexBuffer;
    pangolin::GlBuffer colorBuffer;
};

}

