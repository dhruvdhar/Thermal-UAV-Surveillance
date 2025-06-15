# Run inference on test set and crop images
import os
import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
output_base = 'cropped_dataset/train'
os.makedirs(output_base, exist_ok=True)
test_dir = '/tmp/dataset/images/test'
class_names = model.names

for cls_name in class_names.values():
    os.makedirs(os.path.join(output_base, cls_name), exist_ok=True)

for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    img = cv2.imread(img_path)
    results = model(img)

    for i, (box, cls_id) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[int(cls_id)]
        cropped = img[y1:y2, x1:x2]
        save_path = f"{output_base}/{class_name}/{img_file.split('.')[0]}_crop{i}.jpg"
        cv2.imwrite(save_path, cropped)
