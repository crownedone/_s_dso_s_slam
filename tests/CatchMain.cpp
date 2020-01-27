#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

#include <opencv2/core/ocl.hpp>

int main(int argc, char* argv[])
{
    auto c = cv::ocl::Context::getDefault();
    auto q = cv::ocl::Queue::getDefault();

    int success = 0;
    std::thread test([=, &success]()
    {
        // within a thread, queue is new -> override with default queue.
        cv::ocl::Queue::getDefault() = q;
        cv::ocl::Context::getDefault() = c;

        success = Catch::Session().run(argc, argv);
    });

    if (test.joinable())
    {
        test.join();
    }

    return success;
}