#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory>

#include <chrono>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "onnxruntime_cxx_api.h"

using namespace cv;
int load(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }
    Mat image;
    image = imread( argv[1], IMREAD_COLOR );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", image);
    waitKey(0);
    return 0;
}

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

    // Load the Torch model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }


    cv::Mat img = cv::imread(argv[2], cv::IMREAD_COLOR);
    torch::Tensor tensor_image = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kFloat);
    tensor_image = tensor_image.permute({0, 3, 1, 2});

    tensor_image = tensor_image.permute({0, 2, 3, 1});
    cv::Mat img_out(tensor_image.size(1), tensor_image.size(2), CV_8UC3, tensor_image.data_ptr<float>());
    // cv::cvtColor(img_out, img_out, cv::COLOR_BGR2RGB);

    imwrite("temp_input1.jpg", img_out);


    // Load the input image using OpenCV
    cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);

        // Convert to float and normalize
    // cv::Size size(64, 64);  // define the new size of the image
    // cv::resize(image, image, size);  // resize the image
    imwrite("original_input.jpg", image);
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);  // Convert to float and scale to [0, 1]

    // cv::Scalar mean = {0.485, 0.456, 0.406};  // Define mean values
    // cv::Scalar std = {0.229, 0.224, 0.225};  // Define standard deviation values

    // cv::subtract(image, mean, image);  // Subtract mean values from each channel
    // cv::divide(image, std, image);  // Divide each channel by standard deviation values


    // // cv::Mat image;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Convert the input image to a Torch tensor
    torch::Tensor input_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat);
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    
    // torch::Tensor temp_input_tensor = input_tensor.permute({0, 2, 3, 1});
    // cv::Mat temp_input_image(temp_input_tensor.size(1), temp_input_tensor.size(2), CV_32FC3, temp_input_tensor.to(torch::kCPU).data_ptr<float>());
    // cv::multiply(temp_input_image, std, temp_input_image);  // Multiply each channel by standard deviation values
    // cv::add(temp_input_image, mean, temp_input_image);  // Add mean values to each channel
    // cv::cvtColor(temp_input_image, temp_input_image, cv::COLOR_BGR2RGB);
    // temp_input_image *= 255.0;

    // imwrite("temp_input.jpg", temp_input_image);
    // std::cout << "Shape after permutation: " << input_tensor.size(0) << " " <<input_tensor.size(1) << " "<< input_tensor.size(2) << " "<<  input_tensor.size(3) << std::endl;


    input_tensor = input_tensor.to(torch::kCUDA);

    // Run inference on the input tensor
    
    at::Tensor output_tensor = module.forward({input_tensor}).toTensor();
    std::cout << "Shape after permutation: " << output_tensor.size(0) << " " <<output_tensor.size(1) << " "<< output_tensor.size(2) << " "<<  output_tensor.size(3) << std::endl;

    output_tensor = output_tensor.to(torch::kCPU);
    
    output_tensor = output_tensor.permute({0, 2, 3, 1});
    // std::cout << "Shape after permutation: " << output_tensor.size(0) << " " <<output_tensor.size(1) << " "<< output_tensor.size(2) << " "<<  output_tensor.size(3) << std::endl;


    // // // Convert the output tensor to a cv::Mat object
    cv::Mat output_image(output_tensor.size(1), output_tensor.size(2), CV_32FC3, output_tensor.data_ptr<float>());
    // output_image *= 255.0;

    // cv::multiply(output_image, std, output_image);  // Multiply each channel by standard deviation values
    // cv::add(output_image, mean, output_image);  // Add mean values to each channel

    cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
    output_image *= 255.0;

    imwrite("output.jpg", output_image);


  std::cout << "ok\n";
}

