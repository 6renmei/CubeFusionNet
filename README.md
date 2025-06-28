### CubeFusionNet

This project is an object detection network based on the YOLO11 framework, specifically developed for cube detection.

#### Installing
```
conda create -n cube python=3.8 -y

conda activate cube
pip install -r requirements.txt
```

#### Using
##### To run inference on a single image:
```bash
python scripts/predict_on_an_image.py
```
This script loads the model from weights/best.pt, performs inference, and prints the detection results. The results will be saved in runs/detect/predict_single.


##### To run inference on all images in a folder:
```bash
python scripts/predict_on_a_folder.py
```
This script loads the model from weights/best.pt, performs inference, and prints the detection results. The results will be saved in runs/detect/predict_folder.

##### To train the model:
```bash
python scripts/train.py
```

Note: If you want to customize the model or dataset path, or adjust training parameters like epochs or batch size, modify the corresponding lines in scripts/train.py:
```bash
model = YOLO("cfg/models/CubeFusionNet-MobileNetv4.yaml")  # Path to your model config
results = model.train(
    data="cfg/datasets/train_cube_solid.yaml",  # Path to your dataset config
    epochs=1,          # Number of training epochs
    imgsz=640,         # Image resolution
    batch=128,         # Batch size
    device='2'         # Device ID, use 'cpu' if no GPU
)
```
Training results, logs, and model weights will be saved in runs/detect/train.


##### To evaluate the trained model:
```bash
python scripts/val.py
```
The results will be saved in runs/detect/val.

#### Datasets
Dataset Notes: If your dataset YAML uses relative paths, make sure to set datasets_dir in the Ultralytics settings at runtime:
```bash
from ultralytics.utils import SETTINGS
SETTINGS['datasets_dir'] = os.getcwd()
```
If using absolute paths, this is not necessary.


Dataset Format:
[Ultralytics YOLO](https://docs.ultralytics.com/zh/datasets/detect/)