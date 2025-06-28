import sys
import os
import torch

project_root = os.path.abspath(os.getcwd())  # project root directory
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO
from PIL import Image

# 1. 加载训练好的模型
model = YOLO("weights/best.pt")  # 修改为你的模型路径，比如 'runs/detect/train/weights/best.pt'
model.to('cuda:1')

# 2. 设置输入图片文件夹路径
image_dir = "datasets/Official_Dataset/images"  # 替换为你的文件夹路径，比如 "images/val"
# output_dir = "/home/rjm/my_project/ultralytics/Official_Dataset"  # 预测结果保存路径

# os.makedirs(output_dir, exist_ok=True)

# 3. 遍历文件夹中的所有图片文件
for filename in os.listdir(image_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        image_path = os.path.join(image_dir, filename)

        # 4. 执行预测
        results = model(image_path, save=True, save_txt=True, project="runs/detect", name='predict_folder', exist_ok=True)

        # 5. 提取并打印结果
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # 获取边框 [x1, y1, x2, y2]
        confs = result.boxes.conf.cpu().numpy()  # 置信度
        classes = result.boxes.cls.cpu().numpy()  # 类别索引
        names = [result.names[int(cls)] for cls in classes]  # 类别名称
        print(f"✅ {filename} - 检测到 {len(boxes)} 个目标：{names}")
