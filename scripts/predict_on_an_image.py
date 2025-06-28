import torch
import sys
import os

project_root = os.path.abspath(os.getcwd())  # project root directory
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO

# 1. 加载训练好的模型
model = YOLO("weights/best.pt")  # 修改为你的模型路径
model.to('cuda:1')  # 使用 GPU 1

# 2. 设置单张图片路径
image_path = "datasets/Official_Dataset/images/1.png"  # 你要检测的单张图片路径

# 3. 执行预测
results = model(image_path, save=True, save_txt=True, project="runs/detect", name='predict_single', exist_ok=True)

# 4. 提取并打印结果
result = results[0]
boxes = result.boxes.xyxy.cpu().numpy()  # 边框坐标
confs = result.boxes.conf.cpu().numpy()  # 置信度
classes = result.boxes.cls.cpu().numpy()  # 类别索引
names = [result.names[int(cls)] for cls in classes]  # 类别名称

print(f"✅ {image_path} - 检测到 {len(boxes)} 个目标：{names}")
