from ultralytics import YOLO
if __name__ == '__main__':
    # build.YOLODataset = YOLOWeightedDataset
    model = YOLO("yolov9s-multi.yaml").load("yolov9s.pt")  # build from YAML and transfer weights
    
    results = model.train(
        data='C:/Users/dians/Desktop/ultralytics-main-multi/ultralytics/cfg/mycfg/my-multi.yaml', epochs=120,
        batch=1, imgsz=640, nbs=2)
    
