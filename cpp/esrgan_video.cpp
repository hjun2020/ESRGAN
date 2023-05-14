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
#include <vector>
using namespace cv;
using namespace std;
static torch::Tensor input_im_buffer[5];
// static torch::Tensor net_output_buffer[5];
static void * cap;
static int buff_index = 0;
static cv::Mat output_im_buffer[5];


static vector<vector<torch::Tensor>> tensorArray;
static vector<vector<cv::Mat>> net_output_buffer;
static vector<vector<vector<Mat>>> net_output_vector;


static int rows;
static int cols;

static vector<int> row_pts;
static vector<int> col_pts;

static int demo_done = 1;


static torch::jit::script::Module module;

std::vector<int> divideInteger(int n, int d) {
    std::vector<int> parts(d+1, n / d);  // Initialize all parts with the equal quotient
    parts[0] = 0;
    int remainder = n % d;  // Calculate the remainder

    // Distribute the remainder among the parts
    for (int i = 0; i < remainder; ++i) {
        parts[i+1]++;
    }

    for (int i = 1; i < d+1; ++i) {
        parts[i] += parts[i-1];
    }

    return parts;
}

static cv::Mat reassembleGrid(const std::vector<std::vector<cv::Mat>>& grid) {
    // Create a vector to store the rows of the reassembled image
    std::vector<cv::Mat> rows;

    // Iterate over each row in the grid
    for (const auto& row : grid) {
        // Concatenate the images horizontally to form a row
        cv::Mat rowImage;
        cv::hconcat(row, rowImage);
        rows.push_back(rowImage);
    }

    // Concatenate the rows vertically to obtain the reassembled image
    cv::Mat reassembledImage;
    cv::vconcat(rows, reassembledImage);

    return reassembledImage;
}


static Mat get_mat_from_stream(void *p)
{
    VideoCapture *cap = (VideoCapture *)p;
    Mat m;
    *cap >> m;
    if(m.empty()){
      demo_done=0;
      Mat a;
     return a;
    }
      
    return m;
}

static void load_input_mat_demo()
{
    cv::Mat image= get_mat_from_stream(cap);
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);  // Convert to float and scale to [0, 1]

    if(demo_done ==0) return;
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // Convert the input image to a Torch tensor
    torch::Tensor input_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kFloat);

    input_tensor = input_tensor.permute({0, 3, 1, 2});
    for(int i=0; i < rows*cols; i++){
      int r = i / cols + 1;
      int c = i % cols + 1;
      torch::Tensor input_tensor1 = input_tensor.slice(2, row_pts[r-1], row_pts[r]).slice(3, col_pts[c-1], col_pts[c]).to(torch::kCUDA);
      tensorArray[buff_index%5][i] = input_tensor1;
    }
}

static void convert_output_tensor_demo(int count)
{
    Mat concatenatedImage;

    concatenatedImage = reassembleGrid(net_output_vector[(buff_index+2)%5]);

    cv::cvtColor(concatenatedImage, concatenatedImage, cv::COLOR_BGR2RGB);
    concatenatedImage *= 255.0;
    // std::stringstream ss;
    // ss << "image" << count << ".jpg";
    // std::string filename = ss.str();
    // cv::imwrite(filename, concatenatedImage);

    // cout << concatenatedImage.rows << " " << concatenatedImage.cols <<  endl;
    
    output_im_buffer[(buff_index+2)%5] = concatenatedImage.clone(); 

}

static void inference_demo()
{

    for(int i=0; i<rows*cols; i++){
      int r = i / cols;
      int c = i % cols;

      torch::Tensor output = module.forward({tensorArray[(buff_index+1)%5][i]}).toTensor();
      output = output.to(torch::kCPU);
      output = output.permute({0, 2, 3, 1});
      cv::Mat output_image(output.size(1), output.size(2), CV_32FC3, output.data_ptr<float>());
      net_output_vector[(buff_index+1)%5][r][c] =  output_image.clone();
      // net_output_buffer[(buff_index+1)%5][i] = output_image.clone();
    }

    
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

static const size_t num = 3;

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
    // Create the output video writer
    // cv::VideoCapture cap1(argv[2]);
    int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    VideoCapture temp_cap = *(VideoCapture *) cap;
    int fps = temp_cap.get(cv::CAP_PROP_FPS);
    int img_width = temp_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int img_height = temp_cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    rows = 2;
    cols = 4;

    col_pts = divideInteger(img_width, cols);
    row_pts = divideInteger(img_height, rows);


    tensorArray.resize(5, vector<torch::Tensor>(rows*cols));
    net_output_buffer.resize(5, vector<Mat>(rows*cols));
    net_output_vector.resize(5, vector<vector<Mat>>(rows, vector<Mat>(cols)));


    cout << img_height << " " <<  img_width << endl;



    load_input_mat_demo();
    for (size_t i = 1; i < tensorArray.size(); ++i) {
        std::copy(tensorArray[0].begin(), tensorArray[0].end(), tensorArray[i].begin());
    }

    for(int i=0; i<5; i++){
      inference_demo();
      buff_index = (buff_index+1)%5;
    }

    for(int i=0; i<5; i++){
      convert_output_tensor_demo(i);
      buff_index = (buff_index+1)%5;
    }

    auto start_time = std::chrono::high_resolution_clock::now();  
    void *ptr;
    int count = 0;

    cv::VideoWriter writer("output_video.mp4", fourcc, fps, cv::Size(img_width*4, img_height*4));

    std::chrono::duration<int, std::milli> sleep_duration(100);
    thread greeters[num];
    while(demo_done){
        greeters[0] = thread(convert_output_tensor_demo, count);
        greeters[1] = thread(load_input_mat_demo);
        greeters[2] = thread(inference_demo);

        // Wait for threads to finish
        for(thread &greeter: greeters){
          greeter.join();
        }
        // if (count > 5) {
        cv::Mat img_8u;
        output_im_buffer[(buff_index+1)%5].convertTo(img_8u, CV_8U);
        writer.write(img_8u);
        // }
         

        buff_index = (buff_index+1)%5;
        count++;

        // if(count>50) break;
        
        if(count % 100 == 0) cout << "current frame is " <<  count << endl;
    }
    writer.release();

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Elapsed time: " << elapsed_time.count() << "seconds" << std::endl;




  std::cout << "ok\n";
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
// // Concatenate the images horizontally
// cv::hconcat(net_output_buffer[(buff_index+2)%5][0], net_output_buffer[(buff_index+2)%5][1], concatenatedImage);
// cv::cvtColor(concatenatedImage, concatenatedImage, cv::COLOR_BGR2RGB);
// concatenatedImage *= 255.0;



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