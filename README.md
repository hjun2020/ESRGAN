## ESRGAN(Enhanced SRGAN)-Video application  [:rocket: [BasicSR](https://github.com/xinntao/BasicSR)] [[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)]

:sparkles: **Notice.**
This project makes use of the architecture and pretrained model provided by Real-ESRGAN(https://github.com/xinntao/Real-ESRGAN). Using the model's architecture and pretrained weights, I converted it into a C++ Torch module using TorchScript and applied it to the video file. It processes a frame approximately a second, 1 fps(frame per seconds). 
I made a demo video on Youtube, https://www.youtube.com/watch?v=vhp8mT9sG5o.



## Quick Test
#### Dependencies
- Python 3
- Libtorch 
- CUDA version >= 11.0
- opcvcv >= 3.0


### Test models
1. Clone this github repo.
```
git clone https://github.com/hjun2020/ESRGAN
cd ESRGAN/cpp
mkdir build
cd build

```
2. in `build` folder:
```
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release
```
2. Place your own **low-resolution .mp4 video file ** in `./LR` folder.
3. Download pretrained models from [Google Drive](https://drive.google.com/file/d/16wH3wcOle33Dld4CXMIuupccq5nbSv-p/view?usp=sharing).
4. Run test. In `build` folder, 
```
./esrgan_video ../../models/generator_ex_gpu.pt ../../LR/{your_video_file}.mp4
```
5. The results are in `./cpp/build` folder.
