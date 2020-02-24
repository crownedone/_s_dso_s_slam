#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

#include <opencv2/core/ocl.hpp>
#include <Logging.hpp>

int main(int argc, char* argv[])
{
    cv::ocl::Context context;

    if (!context.create(cv::ocl::Device::TYPE_GPU))
    {
        LOG_WARNING("Failed creating the context...");
    }

    LOG_INFO( "Detected %d GPU devices.", context.ndevices());

    for (int i = 0; i < context.ndevices(); i++)
    {
        cv::ocl::Device device = context.device(i);
        LOG_INFO("\nname: %s\navailable: %d\nOpenCL_C_Version: %s\n", device.name().c_str(),
                 device.available(),
                 device.OpenCL_C_Version().c_str());
    }

    // Select the first device
    cv::ocl::Device d(context.device(0));

    auto c = cv::ocl::Context::getDefault();
    auto q = cv::ocl::Queue::getDefault();

    int success = 0;
    std::thread test([ =, &success]()
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