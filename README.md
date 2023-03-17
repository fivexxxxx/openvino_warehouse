
## demo 1:  人脸识别：  



![image](https://github.com/fivexxxxx/openvino_warehouse/blob/master/gif/openvino-face-detection.gif)  



## demo 2:  头部姿态：  




![image](https://github.com/fivexxxxx/openvino_warehouse/blob/master/gif/emotions2.gif)   


持续不定期更新...  


# openvino_warehouse #
openVINO  基于视觉推断与神经网络优化，提供的深度学习推理套件，可以将各种开源框架训练好的模型进行线上部署，比如tensorflow，pytorch，paddlepaddle, onnx，mxnet，caffe2等。当模型训练结束后，上线部署时，就会遇到各种问题，比如，模型性能是否满足线上要求，模型如何嵌入到原有工程系统，推理线程的并发路数是否满足，这些问题决定着投入产出比。只有深入且准确的理解深度学习框架，才能更好的完成这些任务，满足上线要求。实际情况是，新的算法模型和所用框架在不停的变化，这个时候恨不得工程师什么框架都熟练掌握，令人失望的是，这种人才目前是稀缺的。

OpenVINO是一个Pipeline工具集，同时可以兼容各种开源框架训练好的模型，拥有算法模型上线部署的各种能力，只要掌握了该工具，你可以轻松的将预训练模型在Intel的CPU上快速部署起来。  

## 环境搭建：  

系统：win10 ;OpenVINO 2022.3;python 3.8;pycharm2019.2  



## 建议：  

使用Anaconda搭建环境，方便版本切换，下载地址：https://www.anaconda.com/  

## 命令行方式创建环境：  

    conda create -n openvino-dev python=3.8  

## 查看创建的环境：  

    conda env list  

或  

    conda info -e  

## 激活环境：  

    activate openvino-dev  

## 升级PIP：  

    python -m pip install --upgrade pip  

## OpenVINO 官网下载地址：

https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html  

根据需要在页面中选择即可，最后复制命令行到上面虚拟环境命令行安装即可。

## python代码集成编辑工具  

本仓使用pycharm2019.2社区版，在pycharm里添加上面的openvino环境的步骤如下：  

打开pycharm后设置步骤如下：  

    file->setting->project:face_detection(:项目名称)->project interpreter->找到你环境所在位置的python.exe  
      

