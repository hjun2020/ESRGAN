## ESRGAN(Enhanced SRGAN)-Video application  [:rocket: [BasicSR](https://github.com/xinntao/BasicSR)] [[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)]

:sparkles: **Notice.**
This project makes use of the architecture and pretrained model provided by Real-ESRGAN(https://github.com/xinntao/Real-ESRGAN). Using the model's architecture and pretrained weights, I converted it into a C++ Torch module and applied it to the video file.



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
2. Place your own **low-resolution images** in `./LR` folder. (There are two sample images - baboon and comic).
3. Download pretrained models from [Google Drive](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY) or [Baidu Drive](https://pan.baidu.com/s/1-Lh6ma-wXzfH8NqeBtPaFQ). Place the models in `./models`. We provide two models with high perceptual quality and high PSNR performance (see [model list](https://github.com/xinntao/ESRGAN/tree/master/models)).
4. Run test. We provide ESRGAN model and RRDB_PSNR model and you can config in the `test.py`.
```
python test.py
```
5. The results are in `./results` folder.
