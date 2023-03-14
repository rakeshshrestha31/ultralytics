#include <iostream>
#include <vector>
#include <getopt.h>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    // std::string projectBasePath = "/home/paperspace/raddev/ultralytics"; // Set your ultralytics base path

    if (argc < 6)
    {
        std::cout << "Usage: " << argv[0] << " " << "<model> <classes> <image_dir> <width> <height>" << std::endl;
        exit(1);
    }
    std::string modelFile = std::string(argv[1]);
    std::string classesFile = std::string(argv[2]);
    std::string imageDir = std::string(argv[3]);
    int width = std::stoi(argv[4]);
    int height = std::stoi(argv[5]);

    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(
        modelFile, // projectBasePath + "/yolov8s.onnx",
        cv::Size(width, height),
        classesFile, // "classes.txt",
        runOnGPU
    );

    // std::vector<std::string> imageNames;
    // imageNames.push_back(projectBasePath + "/ultralytics/assets/bus.jpg");
    // imageNames.push_back(projectBasePath + "/ultralytics/assets/zidane.jpg");

    std::filesystem::path imageDirPath(imageDir);

    for (const auto& entry : std::filesystem::directory_iterator(imageDirPath))
    // for (int i = 0; i < imageNames.size(); ++i)
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        auto imageName = entry.path();
        // auto imageName = imageNames[i]

        cv::Mat frame = cv::imread(imageName);

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        // cv::imshow("Inference", frame);

        // cv::waitKey(-1);

        auto outfile = std::string("/tmp/") + std::string(entry.path().stem()) + ".jpg";
        std::cout << outfile << std::endl;
        cv::imwrite(outfile, frame);
    }
}
