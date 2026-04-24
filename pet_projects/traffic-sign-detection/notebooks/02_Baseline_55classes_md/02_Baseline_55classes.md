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
DATASET_PATH = Path("../data/baseline.yaml")
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
    name="baseline_55class",      # Experiment name
    exist_ok=True,                # Overwrite existing folder
    verbose=True,                 # Print training logs
    seed=42,                       # Print training logs
    close_mosaic=50,
    copy_paste=0.5
    
)

print("Training completed!")
```

    Ultralytics 8.4.41  Python-3.12.12 torch-2.2.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 8188MiB)
    [34m[1mengine\trainer: [0magnostic_nms=False, amp=True, angle=1.0, augment=True, auto_augment=randaugment, batch=-1, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=50, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.5, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=..\data\baseline.yaml, degrees=0.0, deterministic=True, device=0, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=100, erasing=0.4, exist_ok=True, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolo11m.pt, momentum=0.937, mosaic=1.0, multi_scale=0.0, name=baseline_55class, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=50, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=C:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=42, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
    Overriding model.yaml nc=80 with nc=55
    
                       from  n    params  module                                       arguments                     
      0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
      1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
      2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
      3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
      4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
      5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
      6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
      7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
      8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
      9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
     10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
     11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
     14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
     15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
     17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
     18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
     20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
     21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
     22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
     23        [16, 19, 22]  1   1453429  ultralytics.nn.modules.head.Detect           [55, 16, None, [256, 512, 512]]
    YOLO11m summary: 232 layers, 20,095,413 parameters, 20,095,397 gradients, 68.4 GFLOPs
    
    Transferred 643/649 items from pretrained weights
    Freezing layer 'model.23.dfl.conv.weight'
    [34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks...
    [34m[1mAMP: [0mchecks passed 
    [34m[1mtrain: [0mFast image access  (ping: 0.00.0 ms, read: 664.6238.0 MB/s, size: 65.2 KB)
    [K[34m[1mtrain: [0mScanning C:\Users\isazo\OneDrive\Рабочий стол\courses\partfolio\pet_projects\traffic-sign-detection\data\raw\Traffic Signs\train\labels.cache... 1956 images, 93 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1956/1956  0.0s
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    [34m[1mAutoBatch: [0mComputing optimal batch size for imgsz=640 at 60.0% CUDA memory utilization.
    [34m[1mAutoBatch: [0mCUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU) 8.00G total, 0.37G reserved, 0.19G allocated, 7.43G free
          Params      GFLOPs  GPU_mem (GB)  forward (ms) backward (ms)                   input                  output
        20095413       68.42         1.875         50.57           nan        (1, 3, 640, 640)                    list
        20095413       136.8         3.532         58.02           nan        (2, 3, 640, 640)                    list
        20095413       273.7         5.918         75.34           nan        (4, 3, 640, 640)                    list
        20095413       547.4        11.239           135           nan        (8, 3, 640, 640)                    list
        20095413        1095        20.189          2851           nan       (16, 3, 640, 640)                    list
    [34m[1mAutoBatch: [0mUsing batch-size 2 for CUDA:0 3.90G/8.00G (49%) 
    [34m[1mtrain: [0mFast image access  (ping: 0.00.0 ms, read: 551.0202.7 MB/s, size: 61.4 KB)
    [K[34m[1mtrain: [0mScanning C:\Users\isazo\OneDrive\Рабочий стол\courses\partfolio\pet_projects\traffic-sign-detection\data\raw\Traffic Signs\train\labels.cache... 1956 images, 93 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1956/1956  0.0s
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    [34m[1mval: [0mFast image access  (ping: 0.00.0 ms, read: 462.2230.8 MB/s, size: 61.0 KB)
    [K[34m[1mval: [0mScanning C:\Users\isazo\OneDrive\Рабочий стол\courses\partfolio\pet_projects\traffic-sign-detection\data\raw\Traffic Signs\valid\labels.cache... 882 images, 43 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 882/882  0.0s
    [34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
    [34m[1moptimizer:[0m AdamW(lr=0.000169, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
    Plotting labels to C:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class\labels.jpg... 
    

    2026/04/24 12:20:33 INFO mlflow.tracking.fluent: Experiment with name '/Shared/Ultralytics' does not exist. Creating a new experiment.
    

    [34m[1mMLflow: [0mlogging run_id(f3d30db307b64c808609edf03602ff75) to ..\models\mlflow
    [34m[1mMLflow: [0mview at http://127.0.0.1:5000 with 'mlflow server --backend-store-uri ..\models\mlflow'
    [34m[1mMLflow: [0mdisable with 'yolo settings mlflow=False'
    Image sizes 640 train, 640 val
    Using 8 dataloader workers
    Logging results to [1mC:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class[0m
    Starting training for 100 epochs...
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      1/100       1.6G     0.7942      4.177      1.001          3        640: 100% ━━━━━━━━━━━━ 978/978 8.8it/s 1:51<0.1sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.7it/s 15.0s<0.1s
                       all        882       1393      0.574      0.314      0.275      0.231
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      2/100       1.6G     0.7766      2.448     0.9971          9        640: 100% ━━━━━━━━━━━━ 978/978 8.4it/s 1:57<0.3sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.6it/s 15.2s<0.1s
                       all        882       1393      0.405       0.46      0.418      0.347
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      3/100       1.6G     0.7817      1.875      1.007          4        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.4it/s 15.4s<0.1s
                       all        882       1393      0.645      0.461      0.476      0.397
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      4/100       1.6G     0.7849       1.78       1.01          7        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.9it/s 14.8s<0.1s
                       all        882       1393      0.673      0.564      0.599      0.497
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      5/100       1.6G     0.7792      1.509      1.018          2        640: 100% ━━━━━━━━━━━━ 978/978 8.5it/s 1:55<0.2ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.4it/s 15.3s<0.1s
                       all        882       1393      0.636      0.583      0.596        0.5
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      6/100       1.6G     0.7842      1.386      1.007          6        640: 100% ━━━━━━━━━━━━ 978/978 8.5it/s 1:55<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.7it/s 15.0s0.1s
                       all        882       1393      0.581      0.574      0.607      0.512
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      7/100       1.6G     0.7525      1.244          1          6        640: 100% ━━━━━━━━━━━━ 978/978 7.7it/s 2:06<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.3it/s 15.4s0.1s
                       all        882       1393      0.718      0.594      0.657      0.558
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      8/100       1.6G     0.7395       1.21     0.9961          9        640: 100% ━━━━━━━━━━━━ 978/978 6.9it/s 2:21<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.3it/s 15.4s0.1s
                       all        882       1393       0.66      0.642      0.682      0.572
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K      9/100       1.6G     0.7407      1.086     0.9978          3        640: 100% ━━━━━━━━━━━━ 978/978 7.3it/s 2:14<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.8it/s 14.9s0.1s
                       all        882       1393      0.806      0.597      0.701      0.592
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     10/100       1.6G     0.7229      1.028     0.9898          3        640: 100% ━━━━━━━━━━━━ 978/978 7.8it/s 2:06<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.7it/s 16.1s<0.1s
                       all        882       1393      0.786      0.672       0.75      0.632
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     11/100       1.6G     0.7263     0.9869     0.9883          2        640: 100% ━━━━━━━━━━━━ 978/978 7.4it/s 2:12<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.0it/s 15.8s<0.1s
                       all        882       1393      0.685      0.745       0.76      0.642
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     12/100       1.6G     0.7277      1.036     0.9985          9        640: 100% ━━━━━━━━━━━━ 978/978 7.4it/s 2:12<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.9it/s 14.8s0.1s
                       all        882       1393       0.79      0.703      0.772      0.658
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     13/100       1.6G     0.7121      1.001     0.9987          3        640: 100% ━━━━━━━━━━━━ 978/978 8.6it/s 1:53<0.1sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.5it/s 14.2s0.1s
                       all        882       1393      0.653      0.709      0.763      0.647
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     14/100       1.6G     0.7225      0.994      1.003          2        640: 100% ━━━━━━━━━━━━ 978/978 7.5it/s 2:10<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.8it/s 14.9s0.1s
                       all        882       1393      0.719      0.686      0.773      0.655
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     15/100       1.6G     0.7126     0.8993     0.9946         12        640: 100% ━━━━━━━━━━━━ 978/978 8.4it/s 1:56<0.1sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.9it/s 15.9s0.1s
                       all        882       1393      0.794      0.709      0.797      0.683
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     16/100       1.6G     0.7025      0.906     0.9932          2        640: 100% ━━━━━━━━━━━━ 978/978 8.4it/s 1:56<0.1sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.5it/s 14.3s<0.1s
                       all        882       1393      0.757       0.76      0.828      0.706
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     17/100       1.6G     0.6934     0.8784     0.9745          0        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:57<0.2ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.4it/s 14.4s0.1s
                       all        882       1393       0.76      0.739      0.815      0.704
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     18/100       1.6G     0.6979     0.8608     0.9873          9        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.5it/s 15.3s0.1s
                       all        882       1393      0.819      0.749       0.83      0.711
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     19/100       1.6G     0.6884     0.7964     0.9713          4        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:60<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.9it/s 14.8s<0.1s
                       all        882       1393      0.824      0.697      0.806      0.689
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     20/100       1.6G     0.6762     0.8048     0.9608          3        640: 100% ━━━━━━━━━━━━ 978/978 7.8it/s 2:06<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.0it/s 17.0s0.2s
                       all        882       1393      0.749      0.771      0.808      0.694
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     21/100       1.6G     0.6682     0.7883     0.9589          3        640: 100% ━━━━━━━━━━━━ 978/978 7.9it/s 2:04<0.3ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.0it/s 15.8s<0.1s
                       all        882       1393      0.744      0.821      0.849      0.728
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     22/100       1.6G     0.6627     0.7659     0.9642          2        640: 100% ━━━━━━━━━━━━ 978/978 7.9it/s 2:04<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.7it/s 15.0s<0.1s
                       all        882       1393      0.812      0.754      0.836      0.718
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     23/100       1.6G     0.6729     0.7857     0.9668          5        640: 100% ━━━━━━━━━━━━ 978/978 8.8it/s 1:52<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.8it/s 15.0s0.1s
                       all        882       1393      0.731      0.792      0.795      0.671
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     24/100       1.6G     0.6975     0.7793     0.9773          1        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 11.9it/s 18.5s0.1ss
                       all        882       1393      0.807      0.813      0.855      0.733
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     25/100       1.6G     0.6664     0.7444      0.968          4        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:00<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.0it/s 14.8s<0.1s
                       all        882       1393      0.829      0.785      0.846      0.724
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     26/100       1.6G     0.6481       0.71     0.9543          6        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:60<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.1it/s 14.6s0.1s
                       all        882       1393      0.788      0.799      0.838       0.72
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     27/100       1.6G     0.6451     0.7275     0.9489          5        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.3it/s 14.4s0.1s
                       all        882       1393      0.752        0.8      0.843      0.729
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     28/100       1.6G     0.6663     0.7226     0.9681          5        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.1it/s 14.7s<0.1s
                       all        882       1393      0.775      0.828      0.841      0.725
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     29/100       1.6G     0.6473     0.6923     0.9529          6        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:60<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.9it/s 14.8s0.1s
                       all        882       1393      0.799      0.791      0.866      0.742
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     30/100       1.6G     0.6548     0.7297     0.9604          8        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.1it/s 14.6s<0.1s
                       all        882       1393      0.786      0.806      0.841      0.725
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     31/100       1.6G     0.6433     0.6759      0.951          2        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.3it/s 14.4s<0.1s
                       all        882       1393       0.81      0.783      0.842      0.731
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     32/100       1.6G     0.6402     0.6969     0.9585          5        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.3it/s 14.4s<0.1s
                       all        882       1393      0.814       0.83      0.862      0.743
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     33/100       1.6G     0.6518     0.6738     0.9546          3        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.7it/s 16.1s<0.1s
                       all        882       1393      0.852      0.812      0.867      0.749
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     34/100       1.6G     0.6416     0.6388     0.9512          1        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.6it/s 15.1s0.1s
                       all        882       1393      0.754       0.85      0.857      0.736
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     35/100       1.6G     0.6419     0.6688     0.9385          3        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.5it/s 15.2s<0.1s
                       all        882       1393       0.86      0.811      0.862      0.747
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     36/100       1.6G     0.6365     0.6682     0.9452          3        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:01<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.9it/s 14.9s0.1s
                       all        882       1393      0.823       0.78      0.842      0.735
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     37/100       1.6G     0.6101     0.6166     0.9275          7        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.2ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.6it/s 15.2s<0.1s
                       all        882       1393      0.823      0.765      0.846      0.737
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     38/100       1.6G     0.6284     0.6232     0.9517          8        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.4it/s 15.4s0.1s
                       all        882       1393      0.771      0.866       0.87      0.754
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     39/100       1.6G     0.6032     0.5953      0.927          7        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.6it/s 14.2s<0.1s
                       all        882       1393      0.822      0.833      0.878      0.759
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     40/100       1.6G     0.6019     0.5912     0.9261          6        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:60<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.1it/s 14.6s<0.1s
                       all        882       1393      0.794      0.838      0.865      0.749
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     41/100       1.6G     0.6162     0.6041     0.9348          3        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.0it/s 14.8s<0.1s
                       all        882       1393      0.863      0.814      0.876      0.763
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     42/100       1.6G     0.6128     0.6273     0.9358          1        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.2ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.4it/s 16.4s0.1s
                       all        882       1393      0.756      0.843      0.866      0.754
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     43/100       1.6G     0.6055     0.5798     0.9299          5        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:57<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.0it/s 14.7s<0.1s
                       all        882       1393      0.845      0.807      0.873      0.766
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     44/100       1.6G     0.6141     0.5923     0.9331         14        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.5it/s 14.2s<0.1s
                       all        882       1393      0.829      0.824      0.871      0.762
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     45/100       1.6G     0.6135     0.5898     0.9329          6        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:60<0.2sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.1it/s 14.6s<0.1s
                       all        882       1393       0.82      0.845      0.882      0.769
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     46/100       1.6G     0.5942     0.5585     0.9324          7        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.2it/s 14.6s<0.1s
                       all        882       1393      0.853      0.816      0.873      0.762
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     47/100       1.6G     0.6091     0.5834     0.9457          4        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:60<0.1s4s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.6it/s 15.1s<0.1s
                       all        882       1393      0.851      0.828       0.87      0.758
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     48/100       1.6G     0.5982     0.6008     0.9423          6        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:57<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.4it/s 14.4s<0.1s
                       all        882       1393      0.825      0.855      0.879      0.767
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     49/100       1.6G     0.5999     0.5773     0.9262          7        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:00<0.2ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.1it/s 14.7s<0.1s
                       all        882       1393      0.826      0.841      0.879      0.771
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     50/100       1.6G     0.6042     0.5759     0.9369          6        640: 100% ━━━━━━━━━━━━ 978/978 7.2it/s 2:16<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.1it/s 14.6s0.1s
                       all        882       1393      0.833      0.835      0.876      0.767
    Closing dataloader mosaic
    [34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     51/100       1.6G      0.559     0.4336     0.9116          3        640: 100% ━━━━━━━━━━━━ 978/978 7.7it/s 2:07<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.5it/s 15.2s<0.1s
                       all        882       1393      0.863      0.835      0.887       0.77
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     52/100       1.6G     0.5566     0.4275     0.9126          2        640: 100% ━━━━━━━━━━━━ 978/978 7.9it/s 2:04<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.9it/s 14.8s0.1s
                       all        882       1393      0.802      0.861      0.883      0.767
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     53/100       1.6G     0.5623     0.4201     0.9067          2        640: 100% ━━━━━━━━━━━━ 978/978 7.7it/s 2:06<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.1it/s 15.7s<0.1s
                       all        882       1393      0.846      0.851      0.884      0.772
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     54/100       1.6G     0.5499     0.4031     0.9069          2        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:60<0.2ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.4it/s 15.3s<0.1s
                       all        882       1393      0.863      0.843      0.891      0.778
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     55/100       1.6G     0.5517     0.4158     0.9101          5        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.2it/s 16.8s<0.1s
                       all        882       1393      0.781      0.876      0.886      0.775
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     56/100       1.6G     0.5367     0.3914     0.9029          3        640: 100% ━━━━━━━━━━━━ 978/978 6.8it/s 2:23<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.0it/s 15.8s<0.1s
                       all        882       1393      0.775       0.86      0.886      0.774
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     57/100       1.6G      0.545     0.3936     0.9028          2        640: 100% ━━━━━━━━━━━━ 978/978 7.5it/s 2:10<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.4it/s 16.6s<0.1s
                       all        882       1393      0.815       0.83      0.893      0.778
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     58/100       1.6G     0.5399     0.3973     0.9013          3        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:60<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.2it/s 14.5s<0.1s
                       all        882       1393      0.843      0.815      0.873      0.764
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     59/100       1.6G     0.5372     0.3914     0.8975          5        640: 100% ━━━━━━━━━━━━ 978/978 7.6it/s 2:09<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.7it/s 15.1s<0.1s
                       all        882       1393      0.861      0.812      0.881      0.775
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     60/100       1.6G     0.5382     0.3845     0.8964          4        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:01<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.9it/s 14.8s0.1s
                       all        882       1393      0.819       0.87      0.898      0.788
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     61/100       1.6G     0.5418     0.3984     0.8964          2        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.6it/s 15.1s<0.1s
                       all        882       1393      0.831      0.853      0.878      0.771
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     62/100       1.6G     0.5225     0.3764     0.8894          2        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:01<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.7it/s 16.1s0.1s
                       all        882       1393      0.869       0.84      0.898      0.783
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     63/100       1.6G     0.5257     0.3689     0.8892          6        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.2ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.6it/s 15.2s<0.1s
                       all        882       1393      0.818      0.853      0.886       0.78
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     64/100       1.6G     0.5224     0.3728     0.8859          2        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:01<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.4it/s 15.3s<0.1s
                       all        882       1393      0.833      0.841      0.877      0.765
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     65/100       1.6G     0.5192       0.36     0.8919          3        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.0it/s 14.8s<0.1s
                       all        882       1393      0.821       0.87       0.89      0.782
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     66/100       1.6G     0.5158     0.3625     0.8899          2        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:01<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.5it/s 15.2s<0.1s
                       all        882       1393      0.874      0.833      0.889      0.782
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     67/100       1.6G     0.5184     0.3677     0.8811          2        640: 100% ━━━━━━━━━━━━ 978/978 8.3it/s 1:58<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.7it/s 15.0s0.1s
                       all        882       1393      0.833      0.864      0.883      0.777
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     68/100       1.6G     0.5195     0.3445     0.8902          4        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:01<0.1sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.7it/s 15.0s<0.1s
                       all        882       1393      0.835      0.853      0.891      0.782
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     69/100       1.6G     0.5166     0.3461     0.8859          4        640: 100% ━━━━━━━━━━━━ 978/978 8.0it/s 2:02<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.1it/s 14.6s<0.1s
                       all        882       1393      0.841      0.843      0.877      0.769
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     70/100       1.6G     0.5136     0.3498     0.8839          3        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:01<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.6it/s 15.2s0.1s
                       all        882       1393      0.818      0.871      0.879      0.776
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     71/100       1.6G     0.5112     0.3386     0.8863          4        640: 100% ━━━━━━━━━━━━ 978/978 7.6it/s 2:09<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.5it/s 15.2s0.1s
                       all        882       1393      0.818      0.858      0.879      0.778
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     72/100       1.6G     0.4965     0.3231     0.8825          3        640: 100% ━━━━━━━━━━━━ 978/978 7.7it/s 2:07<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.5it/s 15.3s0.1s
                       all        882       1393      0.855      0.851      0.899      0.791
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     73/100       1.6G     0.4963     0.3273     0.8759          1        640: 100% ━━━━━━━━━━━━ 978/978 7.5it/s 2:11<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.6it/s 16.2s0.1s
                       all        882       1393      0.818      0.859      0.887      0.778
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     74/100       1.6G     0.4944     0.3292     0.8778          4        640: 100% ━━━━━━━━━━━━ 978/978 7.8it/s 2:06<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.3it/s 15.4s<0.1s
                       all        882       1393      0.819       0.85      0.884      0.774
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     75/100       1.6G      0.494     0.3273     0.8798          6        640: 100% ━━━━━━━━━━━━ 978/978 7.5it/s 2:10<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.2it/s 15.5s0.1s
                       all        882       1393      0.833      0.839      0.885      0.779
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     76/100       1.6G     0.4894     0.3245     0.8761          2        640: 100% ━━━━━━━━━━━━ 978/978 7.7it/s 2:07<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.5it/s 15.2s0.1s
                       all        882       1393      0.846      0.846      0.885       0.78
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     77/100       1.6G     0.4832       0.31     0.8722          4        640: 100% ━━━━━━━━━━━━ 978/978 7.6it/s 2:09<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.4it/s 15.3s0.1s
                       all        882       1393      0.838      0.842      0.888       0.78
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     78/100       1.6G     0.4917      0.319      0.879          5        640: 100% ━━━━━━━━━━━━ 978/978 7.6it/s 2:09<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.9it/s 15.9s0.2s
                       all        882       1393      0.838      0.864      0.895      0.788
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     79/100       1.6G     0.4792     0.3135     0.8783          3        640: 100% ━━━━━━━━━━━━ 978/978 7.5it/s 2:10<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.1it/s 15.6s0.1s
                       all        882       1393      0.831      0.858      0.887       0.78
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     80/100       1.6G     0.4861     0.3224     0.8741          4        640: 100% ━━━━━━━━━━━━ 978/978 7.6it/s 2:09<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.3it/s 15.4s0.1s
                       all        882       1393      0.827      0.871      0.881      0.772
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     81/100       1.6G     0.4798      0.307     0.8703          2        640: 100% ━━━━━━━━━━━━ 978/978 7.5it/s 2:10<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.1it/s 15.7s0.1s
                       all        882       1393      0.825       0.86      0.883      0.777
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     82/100       1.6G     0.4791     0.3022     0.8768          5        640: 100% ━━━━━━━━━━━━ 978/978 7.6it/s 2:08<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.4it/s 15.4s<0.1s
                       all        882       1393      0.845       0.86      0.891      0.782
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     83/100       1.6G     0.4765     0.3195     0.8695          2        640: 100% ━━━━━━━━━━━━ 978/978 7.4it/s 2:11<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.7it/s 16.1s0.1s
                       all        882       1393      0.824      0.858      0.887      0.779
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     84/100       1.6G     0.4743     0.3005      0.869          4        640: 100% ━━━━━━━━━━━━ 978/978 7.4it/s 2:13<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.8it/s 16.0s<0.1s
                       all        882       1393      0.852      0.853      0.891      0.782
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     85/100       1.6G     0.4667     0.2969     0.8655          1        640: 100% ━━━━━━━━━━━━ 978/978 7.5it/s 2:11<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.2it/s 15.6s<0.1s
                       all        882       1393      0.877      0.838      0.893      0.783
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     86/100       1.6G      0.465     0.2942     0.8644          3        640: 100% ━━━━━━━━━━━━ 978/978 7.4it/s 2:12<0.2sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.1it/s 15.7s<0.1s
                       all        882       1393       0.88      0.838      0.893      0.784
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     87/100       1.6G     0.4681     0.2876     0.8636          3        640: 100% ━━━━━━━━━━━━ 978/978 6.8it/s 2:25<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.5it/s 15.3s<0.1s
                       all        882       1393      0.844      0.861      0.899      0.789
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     88/100       1.6G     0.4636      0.296     0.8623          9        640: 100% ━━━━━━━━━━━━ 978/978 8.4it/s 1:57<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.5it/s 15.3s<0.1s
                       all        882       1393       0.86      0.844      0.894      0.785
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     89/100       1.6G     0.4524     0.2797     0.8612          5        640: 100% ━━━━━━━━━━━━ 978/978 7.7it/s 2:08<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 11.1it/s 19.9s0.2s
                       all        882       1393      0.847      0.867      0.888      0.783
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     90/100       1.6G     0.4576     0.2869     0.8622          4        640: 100% ━━━━━━━━━━━━ 978/978 7.2it/s 2:15<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 12.9it/s 17.1s<0.1s
                       all        882       1393      0.862      0.864      0.894      0.786
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     91/100       1.6G     0.4491     0.2756     0.8565          3        640: 100% ━━━━━━━━━━━━ 978/978 6.3it/s 2:36<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.4it/s 16.5s<0.1s
                       all        882       1393      0.849      0.861      0.888       0.78
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     92/100       1.6G     0.4515     0.2787     0.8532          2        640: 100% ━━━━━━━━━━━━ 978/978 7.6it/s 2:09<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.5it/s 14.3s<0.1s
                       all        882       1393      0.871      0.853      0.892       0.78
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     93/100       1.6G     0.4489     0.2679     0.8566          3        640: 100% ━━━━━━━━━━━━ 978/978 8.2it/s 1:59<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 13.6it/s 16.3s0.2s
                       all        882       1393      0.849      0.875      0.896      0.787
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     94/100       1.6G     0.4482     0.2757     0.8587          2        640: 100% ━━━━━━━━━━━━ 978/978 7.5it/s 2:11<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.6it/s 15.2s0.1s
                       all        882       1393       0.86      0.873      0.896      0.788
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     95/100       1.6G      0.456     0.2752     0.8593          6        640: 100% ━━━━━━━━━━━━ 978/978 8.1it/s 2:01<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.1it/s 15.7s0.1s
                       all        882       1393       0.86      0.877      0.902      0.796
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     96/100       1.6G     0.4416     0.2674     0.8584          3        640: 100% ━━━━━━━━━━━━ 978/978 7.7it/s 2:07<0.1sss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.2it/s 15.6s<0.1s
                       all        882       1393       0.86      0.884      0.904      0.793
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     97/100       1.6G     0.4448     0.2771     0.8582          2        640: 100% ━━━━━━━━━━━━ 978/978 7.2it/s 2:16<0.1s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 12.5it/s 17.7s0.2s
                       all        882       1393      0.882      0.874      0.905      0.796
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     98/100       1.6G     0.4377     0.2724     0.8569          4        640: 100% ━━━━━━━━━━━━ 978/978 7.0it/s 2:19<0.2s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.1it/s 14.7s<0.1s
                       all        882       1393      0.853      0.884      0.901      0.792
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K     99/100       1.6G     0.4415     0.2678      0.857          6        640: 100% ━━━━━━━━━━━━ 978/978 7.8it/s 2:06<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 15.0it/s 14.7s0.1s
                       all        882       1393       0.87      0.859      0.905      0.793
    
          Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    [K    100/100       1.6G     0.4385     0.2716     0.8574          3        640: 100% ━━━━━━━━━━━━ 978/978 7.7it/s 2:06<0.1ss
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 14.1it/s 15.7s0.2s
                       all        882       1393      0.859      0.875      0.902      0.792
    
    100 epochs completed in 3.908 hours.
    Optimizer stripped from C:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class\weights\last.pt, 40.6MB
    Optimizer stripped from C:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class\weights\best.pt, 40.6MB
    
    Validating C:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class\weights\best.pt...
    Ultralytics 8.4.41  Python-3.12.12 torch-2.2.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 8188MiB)
    YOLO11m summary (fused): 126 layers, 20,072,437 parameters, 0 gradients, 67.9 GFLOPs
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 221/221 9.1it/s 24.3s<0.1s
                       all        882       1393      0.856      0.891      0.901      0.786
                forb_ahead         45         46      0.937      0.978      0.989      0.871
                 forb_left         12         12      0.842      0.917      0.839      0.734
             forb_overtake         13         14      0.963          1      0.995      0.889
                forb_right         11         11      0.842          1      0.988      0.891
        forb_speed_over_10         11         12          1      0.834      0.995      0.833
       forb_speed_over_100         11         13      0.953      0.923      0.986      0.914
       forb_speed_over_130         14         17      0.924      0.824      0.948      0.824
        forb_speed_over_30         30         30          1      0.939      0.995      0.853
        forb_speed_over_40         11         11      0.776      0.909      0.928      0.829
         forb_speed_over_5         10         10      0.957          1      0.995      0.892
        forb_speed_over_50          9         10      0.954        0.9      0.986      0.872
        forb_speed_over_60         20         21      0.934      0.952      0.973      0.828
        forb_speed_over_70          5          5      0.914        0.8      0.795      0.716
        forb_speed_over_80         17         17      0.757      0.918      0.946      0.859
        forb_speed_over_90          4          4      0.733          1      0.995      0.862
             forb_stopping        115        115      0.984      0.974      0.993      0.852
               forb_trucks          1          1      0.591          1      0.995      0.895
               forb_u_turn          8          8      0.946       0.75      0.806      0.727
     forb_weight_over_3.5t          9          9      0.805      0.921      0.941      0.724
     forb_weight_over_7.5t         10         10          1      0.915      0.995        0.9
          info_bus_station         21         21      0.702      0.905       0.77      0.678
            info_crosswalk        113        125      0.949      0.902      0.977      0.811
              info_highway         16         17      0.801      0.882      0.828      0.569
      info_one_way_traffic         22         23      0.893       0.87      0.895      0.727
              info_parking         25         32      0.818      0.842      0.892      0.697
         info_taxi_parking         10         10          1      0.931      0.995      0.894
            mand_bike_lane          6          6      0.825      0.833      0.835       0.76
                 mand_left          9          9      0.419      0.404      0.303      0.272
           mand_left_right          2          2      0.599          1      0.995      0.746
            mand_pass_left          5          5      0.244        0.2      0.115      0.084
      mand_pass_left_right         40         42      0.981      0.762      0.966      0.807
           mand_pass_right         23         24      0.829      0.608      0.721      0.616
                mand_right         41         42      0.847      0.924      0.864       0.75
           mand_roundabout         87         87          1      0.977      0.995      0.933
         mand_straigh_left          6          6      0.716      0.433      0.574      0.509
             mand_straight         10         10      0.991        0.8      0.884      0.717
       mand_straight_right         11         11      0.722      0.946      0.868      0.774
             prio_give_way        181        184      0.924      0.978      0.958      0.838
        prio_priority_road         87         87      0.985      0.966      0.982      0.891
                 prio_stop         76         76      0.962      0.974      0.971      0.922
             warn_children         14         14       0.94          1      0.995      0.914
         warn_construction         16         16      0.926      0.938      0.935       0.84
            warn_crosswalk         32         32      0.875      0.906      0.911       0.72
             warn_cyclists         20         20      0.761       0.95      0.897      0.812
     warn_domestic_animals          3          3      0.873          1      0.995      0.895
        warn_other_dangers         17         17       0.95      0.882      0.941      0.824
    warn_poor_road_surface         14         15      0.914      0.933      0.924      0.782
           warn_roundabout          8          8       0.48          1      0.949      0.826
        warn_slippery_road          3          3      0.704          1      0.746      0.523
         warn_speed_bumper         13         13      0.882      0.923       0.89      0.826
        warn_traffic_light         21         21      0.983          1      0.995       0.92
                 warn_tram         13         13      0.985          1      0.995      0.895
      warn_two_way_traffic          4          4          1      0.977      0.995       0.97
         warn_wild_animals         18         19      0.938          1      0.995       0.91
    Speed: 0.3ms preprocess, 22.8ms inference, 0.0ms loss, 1.2ms postprocess per image
    Results saved to [1mC:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class[0m
    [34m[1mMLflow: [0mresults logged to ..\models\mlflow
    [34m[1mMLflow: [0mdisable with 'yolo settings mlflow=False'
    Training completed!
    

### Evaluate on validation set


```python
best_model_path = OUTPUT_PATH / "detect/baseline_55class/weights/best.pt"
best_model = YOLO(str(best_model_path))

val_results = best_model.val(
    data=str(DATASET_PATH),      # Dataset configuration
    batch=16,                     # Validation batch size
    imgsz=640,                    # Image size for validation
    device=device,                # GPU or CPU
    verbose=True,                 # Print validation results
    name='baseline_55class_val'
)


```

    Ultralytics 8.4.41  Python-3.12.12 torch-2.2.0+cu118 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 8188MiB)
    YOLO11m summary (fused): 126 layers, 20,072,437 parameters, 0 gradients, 67.9 GFLOPs
    [34m[1mval: [0mFast image access  (ping: 0.00.0 ms, read: 526.3152.8 MB/s, size: 53.1 KB)
    [K[34m[1mval: [0mScanning C:\Users\isazo\OneDrive\Рабочий стол\courses\partfolio\pet_projects\traffic-sign-detection\data\raw\Traffic Signs\valid\labels.cache... 882 images, 43 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 882/882  0.0s
    [K                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 56/56 2.3it/s 23.9s0.4ss
                       all        882       1393      0.882      0.874      0.905      0.796
                forb_ahead         45         46      0.933      0.908      0.978      0.881
                 forb_left         12         12      0.809      0.917      0.774      0.686
             forb_overtake         13         14      0.959          1      0.995       0.88
                forb_right         11         11      0.913      0.957      0.988      0.866
        forb_speed_over_10         11         12          1      0.793      0.995      0.872
       forb_speed_over_100         11         13       0.97      0.923      0.968        0.9
       forb_speed_over_130         14         17       0.92      0.824      0.955      0.873
        forb_speed_over_30         30         30      0.983        0.9      0.987      0.832
        forb_speed_over_40         11         11      0.824      0.909       0.94      0.853
         forb_speed_over_5         10         10      0.964          1      0.995      0.878
        forb_speed_over_50          9         10          1      0.935      0.995       0.87
        forb_speed_over_60         20         21      0.932      0.952      0.978       0.86
        forb_speed_over_70          5          5      0.904        0.8      0.795      0.716
        forb_speed_over_80         17         17      0.881      0.941      0.956      0.876
        forb_speed_over_90          4          4      0.724          1      0.995      0.945
             forb_stopping        115        115      0.986      0.948       0.99      0.835
               forb_trucks          1          1       0.83          1      0.995      0.895
               forb_u_turn          8          8      0.943       0.75      0.745      0.667
     forb_weight_over_3.5t          9          9      0.885      0.859      0.938      0.716
     forb_weight_over_7.5t         10         10          1      0.919      0.995      0.909
          info_bus_station         21         21       0.76      0.857      0.775      0.693
            info_crosswalk        113        125      0.946      0.888      0.967      0.805
              info_highway         16         17      0.825      0.882      0.829       0.59
      info_one_way_traffic         22         23      0.951      0.841      0.907       0.73
              info_parking         25         32      0.965      0.852      0.913       0.71
         info_taxi_parking         10         10          1       0.83      0.995      0.899
            mand_bike_lane          6          6      0.821      0.833      0.835      0.776
                 mand_left          9          9      0.726      0.333      0.403      0.353
           mand_left_right          2          2      0.576          1      0.995      0.795
            mand_pass_left          5          5      0.314        0.2      0.114     0.0815
      mand_pass_left_right         40         42       0.98      0.738      0.962      0.804
           mand_pass_right         23         24      0.817      0.559      0.699      0.601
                mand_right         41         42      0.841      0.879      0.876      0.772
           mand_roundabout         87         87          1       0.96      0.994      0.919
         mand_straigh_left          6          6      0.741      0.333      0.686      0.609
             mand_straight         10         10          1      0.717      0.902      0.738
       mand_straight_right         11         11      0.759      0.909      0.863      0.795
             prio_give_way        181        184      0.835      0.978      0.912      0.789
        prio_priority_road         87         87      0.985      0.966      0.983      0.894
                 prio_stop         76         76      0.987      0.968      0.971      0.918
             warn_children         14         14      0.933          1      0.995      0.914
         warn_construction         16         16          1      0.933      0.935      0.846
            warn_crosswalk         32         32       0.88      0.906      0.928       0.75
             warn_cyclists         20         20      0.758       0.95      0.907      0.811
     warn_domestic_animals          3          3      0.865          1      0.995      0.929
        warn_other_dangers         17         17      0.978      0.882      0.936      0.846
    warn_poor_road_surface         14         15       0.91      0.867      0.918      0.793
           warn_roundabout          8          8       0.61      0.981      0.918      0.824
        warn_slippery_road          3          3      0.694          1      0.913      0.714
         warn_speed_bumper         13         13      0.882      0.923      0.883      0.798
        warn_traffic_light         21         21      0.986          1      0.995      0.905
                 warn_tram         13         13      0.982          1      0.995      0.896
      warn_two_way_traffic          4          4          1      0.998      0.995       0.97
         warn_wild_animals         18         19      0.976          1      0.995      0.899
    Speed: 1.4ms preprocess, 20.1ms inference, 0.0ms loss, 0.9ms postprocess per image
    Results saved to [1mC:\Users\isazo\OneDrive\ \courses\partfolio\pet_projects\traffic-sign-detection\models\detect\baseline_55class_val[0m
    

### Print validation metrics


```python
print("\nValidation Results:")
print(f"mAP50-95: {val_results.box.map:.4f}")
print(f"mAP50: {val_results.box.map50:.4f}")
print(f"Precision: {val_results.box.mp:.4f}")
print(f"Recall: {val_results.box.mr:.4f}")

print(f"\nModel saved to: {OUTPUT_PATH}")
```

    
    Validation Results:
    mAP50-95: 0.7958
    mAP50: 0.9045
    Precision: 0.8823
    Recall: 0.8741
    
    Model saved to: ..\models
    

### Visualization


```python
import sys
sys.path.append('../scripts')  
from visualization import *
%matplotlib inline
```


```python
df = load_results("baseline_55class")
```

    Loaded 100 epochs from baseline_55class
    


```python
plot_losses(df)

```


    
![png](output_21_0.png)
    



```python
plot_map_progress(df)

```


    
![png](output_22_0.png)
    





    (96, 0.79644)




```python
print_summary(df)

```

    
    ==================================================
    TRAINING SUMMARY
    ==================================================
    Best mAP50-95:  0.7964 (epoch 96)
    Final mAP50-95: 0.7923
    Best mAP50:     0.9050
    Best Precision: 0.8824
    Best Recall:    0.8841
    ==================================================
    
     Good! Only 4 epochs after peak
    


```python
plot_confusion_matrix("baseline_55class")
```


    
![png](output_24_0.png)
    



```python
plot_confusion_matrix("baseline_55class_val")
```


    
![png](output_25_0.png)
    


# Analysis of Metrics and Graphs (Baseline, 55 classes)

## Training Dynamics (mAP50-95)
- Rapid growth during the first 10–20 epochs (≈0.23 → 0.7)
- Slower improvement after ~40 epochs
- Plateau after ~60 epochs
- Best result: **0.7964 (epoch 96)**

**Conclusion:**
- The model converges properly  
- Performance is close to the limit of the current setup  
- Further training gives only marginal improvements  

---

## Confusion Matrix

### Overall Pattern
- Strong diagonal indicates good class separation  
- Errors are structured rather than random  

---

### Main Issues

#### 1. Confusion within class groups
- `forb_speed_over_*` — difficult to distinguish (difference only in numbers)
- `mand_*` — confusion between directions (left/right/straight)
- `warn_*`, `info_*` — visually similar signs

**Conclusion:**  
The model learns **category-level features**, but struggles with **fine-grained distinctions**

---

#### 2. Vertical and horizontal patterns
- Some classes are predicted more frequently than others

**Possible causes:**
- class imbalance  
- dominant classes in the dataset  

---

#### 3. Background errors
- False positives and false negatives involving background
- Some objects are classified as background and vice versa

**Possible causes:**
- imbalance between object and background samples  
- small or low-quality objects  

---

## Train vs Validation
- Similar confusion matrices
- Same error patterns

**Conclusion:**
- No strong overfitting  
- Main limitations come from data and class design, not training  

---

## Overall Diagnosis
The model captures **high-level class groups well**,  
but struggles with **fine-grained differences within those groups**

---

## Summary

### Strengths
- Stable convergence
- mAP around 0.8 (solid baseline)
- No significant overfitting
- Good generalization

### Weaknesses
- Confusion between similar classes
- Dataset imbalance
- Limited sensitivity to small details
- Overly fine-grained class definitions




```python

```
