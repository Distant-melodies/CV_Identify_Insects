# CV_Identify_Insects

In this project we build and compare several object detection models for **insect detection in field images**, with a particular focus on:

- A **Faster R-CNN** baseline (two-stage detector)
- Robustness to **image distortions** (noise, blur, dark images, occlusion)
- Behaviour under **class imbalance** and simple **rebalancing**
- **Explainability** via Grad-CAM attention maps
- Comparison with other detection paradigms (YOLOv11x, SS+ResNet34, RealTime-DETR)

The code is organised so that you can train, evaluate, stress-test and explain the detectors in a modular way.

---

## 1. Environment and Dependencies

Tested with:

- Python 3.10+ (3.8–3.11 should also work)
- PyTorch ≥ 2.0
- torchvision ≥ 0.15
- torchmetrics
- pytorch-grad-cam
- OpenCV (optional, for some utils)
- matplotlib

Install the main dependencies (adapt as needed):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics pytorch-grad-cam matplotlib opencv-python
```

---

## 2. Faster R-CNN: Training and Evaluation
From the Faster_R-CNN/ directory:    

```bash
python train.py
python evaluate.py
```

---

## 3. Robustness Experiments (Noise, Blur, Dark, Occlusion)
```bash
# Clean
python evaluate.py --distortion none

# Gaussian noise
python evaluate.py --distortion noise --sigma 0.1
python evaluate.py --distortion noise --sigma 0.2

# Blur
python evaluate.py --distortion blur --sigma 1.0
python evaluate.py --distortion blur --sigma 2.0

# Dark
python evaluate.py --distortion dark --factor 0.5

# Occlusion
python evaluate.py --distortion occlusion --area 0.25

```



---

## 4. Class Imbalance and Rebalancing
In config.py:
```python
USE_IMBALANCED_TRAIN = True

MINORITY_CLASSES = [1, 3, 5]
MINORITY_KEEP_RATIO = 0.2
IMBALANCE_SEED = 42

USE_BALANCED_SAMPLER = True


```

---

## 5. Explainability with Grad-CAM
```bash
python attention_maps_cam.py --ckpt path/to/fasterrcnn_checkpoint.pth
```

## 6. Utilities: Visualising Ground Truth Boxes
```bash
python src/DataProcessing/viz_boxes.py

```


