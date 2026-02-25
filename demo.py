from ultralytics import YOLO
import cv2
import os
import numpy as np

if __name__ == '__main__':
    model = YOLO("C:/Users/dians/Desktop/weights/best.pt")
    directory = r"C:\academic\paper\result_r2\r6"
    image_list = [os.path.join(directory, file) for file in os.listdir(directory) if file.lower().endswith('.jpg')]
    for i, img_path in enumerate(image_list):
        result = model.predict(img_path, retina_masks=True)[0]
        image = cv2.imread(img_path)
        if result.masks is not None:
            boxes = result.boxes
            masks = result.masks
            keypoints = result.keypoints
            for xyxy, cls, conf, mask, keypoint, key_conf in zip(boxes.xyxy, boxes.cls, boxes.conf, masks.xy,
                                                                 keypoints.xy, keypoints.conf):
                xyxy = xyxy.cpu().numpy()
                cls = int(cls.cpu().numpy())
                name = result.names[cls]
                conf = conf.cpu().numpy()
                key_conf = key_conf.cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy.tolist())
                color = (255, 42, 4) if cls == 0 else (255, 219, 11) if cls == 1 else (255, 68, 79)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 10)
                text = f"{cls} {name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.7, 5)
                cv2.rectangle(image, (x1, y1), (x1 + w, y1 - h - 5), color, -1)
                cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2)
                kx, ky = keypoint[0].tolist()
                kx, ky = int(kx), int(ky)
                cv2.circle(image, (kx, ky), 15, (0, 0, 255), -1)
                zero_mask = np.zeros(image.shape, dtype=np.uint8)
                mask = mask.astype(np.int32)
                mask = np.clip(mask, (x1, y1), (x2, y2))
                mask = cv2.fillPoly(zero_mask, [mask], color)
                image = cv2.addWeighted(image, 1, mask, 0.7, 0)
            dir_name, file_name = os.path.split(img_path)
            path = 'C:/Users/dians/Desktop/revise_r5/fig11/'
            os.makedirs(path, exist_ok=True)
            file_name = path + file_name
            cv2.imwrite(file_name, image)
            print(f"Processed: {i+1}/{len(image_list)}")
        else:
            print(f"None detected in image {i+1}")

