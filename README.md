# openvino_warehouse
openVINO  深度学习推理、加速的demo集合  

## 环境搭建：  

系统：win10 ;OpenVINO 2022.3;python 3.8;pycharm2019.2  

OpenVINO 官网下载地址：https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/download.html  


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

## python代码集成编辑工具  

本仓使用pycharm2019.2社区版，在pycharm里添加上面的openvino环境的步骤如下：  

打开pycharm后设置步骤如下：  

    file->setting->project:face_detection(:项目名称)->project interpreter->找到你环境所在位置的python.exe  
      

## demo 1:  人脸识别：  

![image](https://github.com/fivexxxxx/openvino_warehouse/blob/master/gif/openvino-face-detection.gif
)  
