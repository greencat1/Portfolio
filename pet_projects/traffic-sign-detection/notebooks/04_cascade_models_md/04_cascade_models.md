### Imports


```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import yaml
from pathlib import Path
import torch
from ultralytics import YOLO, settings

```

### Paths


```python
DATASET_PATH = Path("../data/5_classes.yaml")
OUTPUT_PATH = Path("../models/")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
```

### Configure runs directory


```python
new_runs_dir = OUTPUT_PATH  
settings.update({'runs_dir': str(new_runs_dir)})
```

### Device selection



```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
```

    Using device: cuda
    

### Load dataset configuration


```python
with open(DATASET_PATH, 'r') as f:
    data_config = yaml.safe_load(f)
print(f"Number of classes: {data_config['nc']}")
```

    Number of classes: 55
    

### Initialize model


```python
model = YOLO("yolo11m.pt")
```

### Train the model



```python
results = model.train(
    data=str(DATASET_PATH),      # Path to dataset YAML
    epochs=100,                   # Number of training epochs
    batch=-1,                     # Auto-batch size (optimizes GPU memory)
    imgsz=640,                    # Input image size
    device=device,                # GPU or CPU
    lr0=0.01,                     # Initial learning rate
    weight_decay=0.0005,          # L2 regularization
    momentum=0.937,               # SGD momentum
    augment=True,                 # Enable data augmentation
    workers=8,                    # Number of data loading workers
    patience=50,                  # Early stopping patience
    save=True,                    # Save checkpoints
    name="cascade_model",      # Experiment name
    exist_ok=True,                # Overwrite existing folder
    verbose=True,                 # Print training logs
    seed=42,                       # Print training logs
    close_mosaic=50,
    copy_paste=0.5
    
)

print("Training completed!")
```

    Ultralytics 8.4.41  Python-3.12.12 torch-2.2.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 8188MiB)
    [34m[1mengine\trainer: [0magnostic_nms=False, amp=True, angle=1.0, augment=True, auto_augment=randaugment, batch=-1, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=..\data\baseline.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=500, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8n.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=baseline_55class, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=50, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=C:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
    Overriding model.yaml nc=80 with nc=55
    
                       from  n    params  module                                       arguments                     
      0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
      1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
      2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
      3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
      4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
      5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
      6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
      7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
      8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
      9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
     10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
     13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
     16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
     17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
     19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
     20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
     22        [15, 18, 21]  1    762037  ultralytics.nn.modules.head.Detect           [55, 16, None, [64, 128, 256]]
    Model summary: 130 layers, 3,021,573 parameters, 3,021,557 gradients, 8.3 GFLOPs
    
    Transferred 319/355 items from pretrained weights
    Freezing layer 'model.22.dfl.conv.weight'
    [34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks...
    [34m[1mAMP: [0mchecks passed 
    [34m[1mtrain: [0mFast image access  (ping: 0.00.0 ms, read: 511.9197.0 MB/s, size: 57.3 KB)
    [K[34m[1mtrain: [0mScanning C:\Users\isazo\OneDrive\Рабочий стол\courses\partfolio\pet_projects\traffic-sign-detection\data\raw\Traffic Signs\train\labels.cache... 1956 images, 93 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1956/1956  0.0s
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    [34m[1mAutoBatch: [0mComputing optimal batch size for imgsz=640 at 60.0% CUDA memory utilization.
    [34m[1mAutoBatch: [0mCUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU) 8.00G total, 0.30G reserved, 0.06G allocated, 7.63G free
          Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
         3021573       8.252         0.663         33.58           nan        (1, 3, 640, 640)                    list
         3021573        16.5         1.065         24.37           nan        (2, 3, 640, 640)                    list
         3021573       33.01         1.573         40.72           nan        (4, 3, 640, 640)                    list
         3021573       66.02         2.609         30.96           nan        (8, 3, 640, 640)                    list
         3021573         132         4.977         73.27           nan       (16, 3, 640, 640)                    list
    [34m[1mAutoBatch: [0mUsing batch-size 16 for CUDA:0 5.31G/8.00G (66%) 
    [34m[1mtrain: [0mFast image access  (ping: 0.00.0 ms, read: 674.8268.5 MB/s, size: 74.0 KB)
    [K[34m[1mtrain: [0mScanning C:\Users\isazo\OneDrive\Рабочий стол\courses\partfolio\pet_projects\traffic-sign-detection\data\raw\Traffic Signs\train\labels.cache... 1956 images, 93 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1956/1956  0.0s
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    [34m[1mval: [0mFast image access  (ping: 0.00.0 ms, read: 295.993.7 MB/s, size: 54.9 KB)
    [K[34m[1mval: [0mScanning C:\Users\isazo\OneDrive\Рабочий стол\courses\partfolio\pet_projects\traffic-sign-detection\data\raw\Traffic Signs\valid\labels.cache... 882 images, 43 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 882/882  0.0s
    [34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
    [34m[1moptimizer:[0m MuSGD(lr=0.01, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
    Plotting labels to C:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class\labels.jpg... 
    

    2026/04/23 23:39:40 INFO mlflow.tracking.fluent: Experiment with name '/Shared/Ultralytics' does not exist. Creating a new experiment.
    

    [34m[1mMLflow: [0mlogging run_id(823fa3347b484c73b15332cb7a1cc2c7) to ..\models\mlflow
    [34m[1mMLflow: [0mview at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri ..\models\mlflow'
    [34m[1mMLflow: [0mdisable with 'yolo settings mlflow=False'
    Image sizes 640 train, 640 val
    Using 8 dataloader workers
    Logging results to [1mC:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class[0m
    Starting training for 500 epochs...
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      1/500       2.3G     0.8489      4.607     0.9699         11        640: 100% ━━━━━━━━━━━━ 123/123 3.9it/s 31.7s0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.1it/s 5.5s0.2s
                       all        882       1393      0.405       0.11     0.0842     0.0711
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      2/500       2.3G     0.9033      3.256     0.9862         10        640: 100% ━━━━━━━━━━━━ 123/123 5.1it/s 24.0s0.3s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.3it/s 5.3s0.2s
                       all        882       1393      0.349      0.355      0.202      0.164
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      3/500       2.3G     0.9325      2.802      1.012          7        640: 100% ━━━━━━━━━━━━ 123/123 5.4it/s 22.8s0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.3it/s 5.3s0.2s
                       all        882       1393      0.239      0.328        0.2      0.158
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      4/500       2.3G     0.9515      2.648      1.029          7        640: 100% ━━━━━━━━━━━━ 123/123 6.2it/s 20.0s0.3s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.1it/s 5.5s0.2s
                       all        882       1393      0.393      0.333      0.265      0.206
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      5/500       2.3G     0.9435      2.459       1.04          7        640: 100% ━━━━━━━━━━━━ 123/123 5.7it/s 21.4s0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 4.7it/s 6.0s0.2s
                       all        882       1393      0.568      0.293      0.329      0.252
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      6/500       2.3G     0.9403      2.228      1.032          5        640: 100% ━━━━━━━━━━━━ 123/123 5.9it/s 20.9s0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.0it/s 5.6s0.2s
                       all        882       1393      0.434      0.448      0.389      0.311
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      7/500       2.3G     0.8825      2.027      1.009         17        640: 100% ━━━━━━━━━━━━ 123/123 5.7it/s 21.6s0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 4.8it/s 5.9s0.2s
                       all        882       1393      0.531      0.439      0.463      0.376
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      8/500       2.3G     0.8729      1.883      1.018         11        640: 100% ━━━━━━━━━━━━ 123/123 5.2it/s 23.6s0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 3.7it/s 7.5s0.3s
                       all        882       1393      0.683      0.387      0.466      0.376
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      9/500       2.3G     0.8615      1.746      1.007         11        640: 100% ━━━━━━━━━━━━ 123/123 5.1it/s 24.2s0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.1it/s 5.5s0.2s
                       all        882       1393       0.59      0.492      0.527      0.429
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     10/500       2.3G     0.8465      1.649      1.011          8        640: 100% ━━━━━━━━━━━━ 123/123 5.5it/s 22.5s0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.1it/s 5.5s0.2s
                       all        882       1393      0.479      0.551      0.533      0.438
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     11/500       2.3G     0.8337      1.523      1.007          9        640: 100% ━━━━━━━━━━━━ 123/123 5.6it/s 22.0s0.3s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.5it/s 5.1s0.2s
                       all        882       1393        0.6      0.477      0.569      0.471
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     12/500       2.3G     0.8243      1.443      1.004         12        640: 100% ━━━━━━━━━━━━ 123/123 5.4it/s 22.9s0.3s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.0it/s 5.6s0.2s
                       all        882       1393      0.573      0.563      0.594      0.485
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     13/500       2.3G     0.8031      1.323     0.9873          9        640: 100% ━━━━━━━━━━━━ 123/123 5.3it/s 23.1s0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 4.9it/s 5.8s0.2s
                       all        882       1393      0.609      0.562      0.605      0.496
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     14/500       2.3G     0.8042      1.286     0.9953          8        640: 100% ━━━━━━━━━━━━ 123/123 5.6it/s 22.0s0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.0it/s 5.6s0.2s
                       all        882       1393      0.741      0.548      0.671      0.559
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     15/500       2.3G     0.8008      1.222     0.9899          8        640: 100% ━━━━━━━━━━━━ 123/123 5.5it/s 22.4s0.3s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.1it/s 5.5s0.2s
                       all        882       1393      0.674      0.612      0.671      0.557
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     16/500       2.3G     0.8081      1.211     0.9863          6        640: 100% ━━━━━━━━━━━━ 123/123 5.7it/s 21.7s0.4s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 5.2it/s 5.4s0.2s
                       all        882       1393      0.694      0.582      0.678      0.566
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     17/500       2.3G     0.7941      1.171     0.9921         10        640: 100% ━━━━━━━━━━━━ 123/123 5.4it/s 22.8s0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 28/28 4.9it/s 5.7s0.2s
                       all        882       1393      0.714      0.592      0.663      0.556
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     18/500       2.3G     0.7806        1.1     0.9804          8        640: 100% ━━━━━━━━━━━━ 123/123 5.5it/s 22.5s0.3s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 89% ━━━━━━━━━━╸─ 25/28 5.5it/s 4.9s<0.5s

### Evaluate on validation set


```python
best_model_path = OUTPUT_PATH / "detect/cascade_model/weights/best.pt"
best_model = YOLO(str(best_model_path))

val_results = best_model.val(
    data=str(DATASET_PATH),      # Dataset configuration
    batch=16,                     # Validation batch size
    imgsz=640,                    # Image size for validation
    device=device,                # GPU or CPU
    verbose=True,                 # Print validation results
)


```

### Print validation metrics


```python
print("\nValidation Results:")
print(f"mAP50-95: {val_results.box.map:.4f}")
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"Precision: {val_results.box.mp:.4f}")
print(f"Recall: {val_results.box.mr:.4f}")

print(f"\nModel saved to: {OUTPUT_PATH}")
```
