import sys
import os

project_root = os.path.abspath(os.getcwd())  # project root directory
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from ultralytics import YOLO

model = YOLO("weights/best.pt")  # 加载模型
model.to('cuda:3')


results = model.val(data="cfg/datasets/cube_test.yaml")  # 评估

# 获取每类指标
precision = results.box.p  # shape: [num_classes]
recall = results.box.r        # shape: [num_classes]
f1 = results.box.f1

# 类别名
names = model.names

# 输出每个类别的指标
for i in range(len(precision)):
    print(f"类别：{names[i]}")
    print(f"   准确率 (Precision): {precision[i]:.4f}")
    print(f"   召回率 (Recall):    {recall[i]:.4f}")
    print(f"   F1 分数 (F1 Score): {f1[i]:.4f}\n")