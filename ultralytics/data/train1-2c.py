#训练代码
from ultralytics import YOLO
if __name__ == '__main__':
    # 然后继续初始化 PyTorch 相关的内容，例如创建模型、加载数据等

    # Load a model
    model = YOLO('/tmp/pycharm_project_455/ultralytics/cfg/models/v8/yolov8m-seg.yaml')
    model = YOLO("/tmp/pycharm_project_455/weight/yolov8m-seg.pt")
    model = YOLO("yolov8m-seg.yaml").load("yolov8m-seg.pt")  # build from YAML and transfer weights
    # Train the model
    results = model.train(
        data='/tmp/pycharm_project_455/ultralytics/cfg/mycfg/tmp-coco8-seg-2c.yaml', epochs=120,
        batch=1, imgsz=(3072, 4096), nbs=1)

