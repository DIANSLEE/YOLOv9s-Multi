# YOLOv9s-Multi
This repository contains the code and dataset for the paper _YOLOv9s-Multi: Orientation-Based Fruit Selection for Robotic Apple Thinning_.  
The OBB (Oriented Bounding Box) comparative experiments are based on MMRotate and Ultralytics YOLO.   
(waiting the final figures)  
# Dataset Download
The dataset, model weight, and code can be downloaded (Baidu): https://pan.baidu.com/s/1EqHlkvU8H5EUnCFpzZUCkg?pwd=chak   
Or Google Drive: https://drive.google.com/drive/folders/1a3eRdFsPp-1PQB6WrNO6_lbgPUYTWyGq?usp=drive_link  
If the link is not work please contact to the author: 1837235434@qq.com
# Notices
Please follow all details by official Ultralytics and MMRotate, this instruction is a simply guiding. If you are trying to set up the environment.
# Installation
This paper code is based on Ultralytics YOLO, readers should using the [official installation](https://docs.ultralytics.com/quickstart/).  
The environment for YOLOv9s-Multi in a Python>=3.10 environment with PyTorch>=2.0.1:
```bash
pip install ultralytics

cd YOLOv9s-Multi
pip install -e .
```
The OBB comparative experiement based on the [MMRotate](https://github.com/open-mmlab/mmrotate?tab=readme-ov-file#installation) and  [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) could be find in their official website. YOLO-OBB same as the paper code settings.  
The environment for MMRotate:
```bash
mmcv-full==1.7.2
python==3.8
mmdet==2.28.2
mmengine==0.10.7
PyTorch>=2.0.1
```
# Running the code
Please put the YOLO dataset as Ultralytics official settings (./ultralytics/data/images and ./ultralytics/data/labels). The directory change to your own path.  
For YOLO, you can use the train.py and changing the config files to change the task.  
```bash
YOLO label format: <class_id> <x_center> <y_center> <width> <height> <keypoint_x> <keypoint_y> <visibility> <segment>
```
```bash
python train.py
```
The standard OBB implement by official code, and the 360 degree variants are implement at [YOLOv9s-OBB-360](https://github.com/DIANSLEE/u2j-obb) and [MMRotate-360](https://github.com/DIANSLEE/mm). The author does not change the running operations, readers should follow the official instructions to re-implement.  
For MMRotate, the config file is in the repository:
```bash
mmrotate/
├── configs/
│   ├── rotated_retinanet/
│   │   └── rotated_retinanet.py
│   └── _base_/schedules/
│       └── schedule_2x.py


mmrotate/data/
          ├── train/
          │   ├── images/
          │   │   ├── 0001.jpg
          │   │   ├── 0002.jpg
          │   │   └── ...
          │   └── annfiles/
          │       └── 0001.txt
          ├── val/
          │   ├── images/
          │   └── annfiles/
          └── test/
              ├── images/
              └── annfiles/
```
```bash
cd mmrotate
python tools/train.py configs/rotated_retinanet/rotated_retinanet.py
```
# Evaluation
For all the model, after training, it can be evaluate by the official evaluation ([YOLO](https://docs.ultralytics.com/modes/val/#usage-examples)).

# Demo
```bash
YOLOv9s-Multi/
├── data/
│ ├── images/ # all images here
│ └── labels/ # corresponding labels here (in the repository)
```
Before running demo.py, change the path to your own.  
We provide best.pt model weight in the dataset link.   
```bash
cd YOLOv9s-Multi
python demo.py
```
Demo result:
![IMG_20230507_084843](https://github.com/user-attachments/assets/5ed80dcb-3b74-4119-af39-9b2d3dd8c227)

# Modification
Copy this:
```bash
# 改动点
```
You can find the all modifications.
