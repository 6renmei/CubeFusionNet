import sys
import os

project_root = os.path.abspath(os.getcwd())  # project root directory
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO


# 加载模型
model = YOLO("cfg/models/CubeFusionNet-MobileNetv4.yaml")  
# model = YOLO("cfg/models/CubeFusionNet.yaml")  
model.info()

# 训练模型
results = model.train(data="cfg/datasets/train_cube_solid.yaml",  # 使用自定义数据集进行训练
                      epochs=1,  # 训练500个周期
                      imgsz=640,  # 调整图像大小
                      batch=128,  # 减少批次大小
                      device='2'  # 使用CPU进行训练
                      )  

print("训练完成")