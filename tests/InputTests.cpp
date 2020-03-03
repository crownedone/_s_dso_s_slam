#include "catch.hpp"
#include "../src/Logging.hpp"
#include "../src/IOWrapper/Input.hpp"


TEST_CASE("Input test ORB_SLAM2", "[Input][RGBD_Tum]")
{
    IO::RGBD_TUM input;

    std::string seq = "C:/Users/steff/Desktop/ORB_SLAM2/Examples/RGB-D/rgbd_dataset_freiburg1_xyz";
    std::string assignment = "C:/Users/steff/Desktop/ORB_SLAM2/Examples/RGB-D/associations/fr1_xyz.txt";

    REQUIRE(input.open(assignment, seq));

    int validFramesSinceStart = 0;
    input.onFrame.connect([ =, &validFramesSinceStart](std::shared_ptr<const IO::FramePack> in)
    {
        if(!in->frame.empty() && !in->depthFrame.empty() && in->timestamp != 0)
        {
            validFramesSinceStart++;
        }
    });

    // start input
    input.playback();
    int seconds = 5;

    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    REQUIRE(validFramesSinceStart > 0);
    LOG_INFO("Got %d images in %d seconds", validFramesSinceStart, seconds);
}

TEST_CASE("Input test DSO", "[Input][Mono_Tum]")
{
    IO::Mono_TUM input;

    std::string seq = "C:/Users/steff/Desktop/dso_seq";

    REQUIRE(input.open(seq));

    int validFramesSinceStart = 0;

    input.onFrame.connect([ =, &validFramesSinceStart](std::shared_ptr<const IO::FramePack> in)
    {
        if (!in->frame.empty() && in->exposure != 0.f && in->timestamp != 0)
        {
            validFramesSinceStart++;
        }
    });

    // start input
    input.playback();
    int seconds = 5;

    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    REQUIRE(validFramesSinceStart > 0);
    LOG_INFO("Got %d images in %d seconds", validFramesSinceStart, seconds);
}