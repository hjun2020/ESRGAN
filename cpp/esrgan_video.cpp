#include <torch/script.h> // One-stop header.
#include <torch/torch.h>

#include <iostream>
#include <memory>

#include <chrono>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "onnxruntime_cxx_api.h"

#include <pthread.h>
#include <thread>
using namespace cv;
using namespace std;
static torch::Tensor input_im_buffer[5];
static torch::Tensor net_output_buffer[5];
static void * cap;
static int buff_index = 0;
static cv::Mat output_im_buffer[5];

static torch::jit::script::Module module;
static cv::Scalar mean_demo = {0.485, 0.456, 0.406};  // Define mean values
static cv::Scalar std_demo = {0.229, 0.224, 0.225};  // Define standard deviation values



static Mat get_mat_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    // if(m.empty()) return make_empty_image(0,0,0);
    return m;
}

static void load_input_mat_demo()
{

    // free(input_im_buffer[buff_index%5]);
    // input_im_buffer[buff_index%3].detach();
    cv::Mat image= get_mat_from_stream(cap);
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);  // Convert to float and scale to [0, 1]
    // cv::Size size(100, 80);  // define the new size of the image
    // cv::resize(image, image, size);  // resize the image
    // cv::subtract(image, mean_demo, image);  // Subtract mean values from each channel
    // cv::divide(image, std_demo, image);  // Divide each channel by standard deviation values

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);


    // Convert the input image to a Torch tensor
    torch::Tensor input_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat);
    input_tensor = input_tensor.permute({0, 3, 1, 2});
    

    input_tensor = input_tensor.to(torch::kCUDA);

    input_im_buffer[buff_index%5] = input_tensor;

}

static void convert_output_tensor_demo(int count)
{
    torch::Tensor output_tensor = net_output_buffer[(buff_index+2)%5];
    output_tensor = output_tensor.permute({0, 2, 3, 1});
    output_tensor = output_tensor.to(torch::kCPU);
    // std::cout << "Shape after permutation: " << output_tensor.size(0) << " " <<output_tensor.size(1) << " "<< output_tensor.size(2) << " "<<  output_tensor.size(3) << std::endl;


    // // // Convert the output tensor to a cv::Mat object
    cv::Mat output_image(output_tensor.size(1), output_tensor.size(2), CV_32FC3, output_tensor.data_ptr<float>());
    // output_image *= 255.0;

    // cv::multiply(output_image, std_demo, output_image);  // Multiply each channel by standard deviation values
    // cv::add(output_image, mean_demo, output_image);  // Add mean values to each channel
    
    cv::Size size(100*4, 80*4);  // define the new size of the image
    cv::resize(output_image, output_image, size);  // resize the image

    cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
    output_image *= 255.0;
    std::stringstream ss;
    ss << "image" << count << ".jpg";
    std::string filename = ss.str();
    cv::imwrite(filename, output_image);
    output_im_buffer[(buff_index+2)%5] = output_image;

}

static void inference_demo()
{

    net_output_buffer[(buff_index+1)%5] = module.forward({input_im_buffer[(buff_index+1)%5]}).toTensor();

}

static void save_demo(int count){
  if (count < 6) return;
  // std::stringstream ss;
  // ss << "image" << count << ".jpg";
  // std::string filename = ss.str();
  // cv::imwrite(filename, output_im_buffer[(buff_index+3)%5]);
}

void *display_in_thread_srcnn_demo(void *args)
{
    cv::imshow("Video", output_im_buffer[(buff_index + 3)%3]);
    // free_image(out_im_buffer[(buff_index + 1)%3]);
    // if (c != -1) c = c%256;
    // if (c == 27) {
    //     demo_done = 1;
    //     return 0;
    // } else if (c == 82) {
    //     demo_thresh += .02;
    // } else if (c == 84) {
    //     demo_thresh -= .02;
    //     if(demo_thresh <= .02) demo_thresh = .02;
    // } else if (c == 83) {
    //     demo_hier += .02;
    // } else if (c == 81) {
    //     demo_hier -= .02;
    //     if(demo_hier <= .0) demo_hier = .0;
    // }
    return 0;
}


void *open_video_stream(const char *f, int c, int w, int h, int fps)
{
    VideoCapture *cap;
    if(f) cap = new VideoCapture(f);
    else cap = new VideoCapture(c);
    if(!cap->isOpened()) return 0;
    if(w) cap->set(CV_CAP_PROP_FRAME_WIDTH, w);
    if(h) cap->set(CV_CAP_PROP_FRAME_HEIGHT, w);
    if(fps) cap->set(CV_CAP_PROP_FPS, w);
    return (void *) cap;
}

static const size_t num = 4;

int main(int argc, const char* argv[]) {
  if (argc != 3) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

    // Load the Torch model
    try {
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    const char *filename = argv[2];
    if(filename){
    printf("video file: %s\n", filename);
    cap = open_video_stream(filename, 0, 0, 0, 0);
    }


    for(int i=0; i<5; i++){
      load_input_mat_demo();
      buff_index = (buff_index+1)%5;
    }

    for(int i=0; i<5; i++){
      inference_demo();
      buff_index = (buff_index+1)%5;
    }

    auto start_time = std::chrono::high_resolution_clock::now();  
    void *ptr;
    int count = 0;
    // Create the output video writer
    cv::VideoCapture cap1(argv[2]);
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    int fps = cap1.get(cv::CAP_PROP_FPS);
    int width = cap1.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap1.get(cv::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter writer("output_video.mp4", fourcc, fps, cv::Size(100*4, 80*4));



    std::chrono::duration<int, std::milli> sleep_duration(100);
    thread greeters[num];
    while(1){
        greeters[0] = thread(convert_output_tensor_demo, count);
        greeters[1] = thread(load_input_mat_demo);
        greeters[2] = thread(inference_demo);
        greeters[3] = thread(save_demo, count);

        // Wait for threads to finish
        for(thread &greeter: greeters){
          greeter.join();
        }

        // std::stringstream ss;
        // ss << "image" << count << ".jpg";
        // std::string filename = ss.str();
        // if (count > 5) cv::imwrite(filename, output_im_buffer[(buff_index+1)%3]);
        if (count > 5) {
          cv::Mat img_8u;
          output_im_buffer[(buff_index+1)%5].convertTo(img_8u, CV_8U);
          writer.write(img_8u);
        }
        

        buff_index = (buff_index+1)%5;
        count++;

        if(count>100) break;
    }
    writer.release();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed_time.count() << "seconds" << std::endl;




  std::cout << "ok\n";
}


    // cv::Mat img = cv::imread("/home/ubuntu/PyTorch-GAN/data/img_align_celeba/000001.jpg", cv::IMREAD_COLOR);
    // torch::Tensor tensor_image = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kFloat);
    // tensor_image = tensor_image.permute({0, 3, 1, 2});

    // tensor_image = tensor_image.permute({0, 2, 3, 1});
    // cv::Mat img_out(tensor_image.size(1), tensor_image.size(2), CV_8UC3, tensor_image.data_ptr<float>());
    // // cv::cvtColor(img_out, img_out, cv::COLOR_BGR2RGB);









    // // Load the input image using OpenCV
    // cv::Mat image = cv::imread("/home/ubuntu/PyTorch-GAN/data/img_align_celeba/000001.jpg", cv::IMREAD_COLOR);

    // imwrite("original_input.jpg", image);
    // image.convertTo(image, CV_32FC3, 1.0 / 255.0);  // Convert to float and scale to [0, 1]

    // cv::subtract(image, mean_demo, image);  // Subtract mean values from each channel
    // cv::divide(image, std_demo, image);  // Divide each channel by standard deviation values


    // cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // // Convert the input image to a Torch tensor
    // torch::Tensor input_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat);
    // input_tensor = input_tensor.permute({0, 3, 1, 2});
    

    // input_tensor = input_tensor.to(torch::kCUDA);

    // // Run inference on the input tensor
    // at::Tensor output_tensor;
    // output_tensor = module.forward({input_tensor}).toTensor();

    // // Stop the timer

    // // Compute the elapsed time


    
    // output_tensor = output_tensor.permute({0, 2, 3, 1});
    // output_tensor = output_tensor.to(torch::kCPU);


    // // Convert the output tensor to a cv::Mat object
    // cv::Mat output_image(output_tensor.size(1), output_tensor.size(2), CV_32FC3, output_tensor.data_ptr<float>());
    // cv::multiply(output_image, std_demo, output_image);  // Multiply each channel by standard deviation values
    // cv::add(output_image, mean_demo, output_image);  // Add mean values to each channel

    // cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
    // output_image *= 255.0;
    // imwrite("temp_input1.jpg", img_out);