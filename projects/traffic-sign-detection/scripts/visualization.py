import random
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from IPython.display import Image as IPImage
import cv2

# Set plot style for professional lookplt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


# ============================================================================
# Random training images with bboxes
# ============================================================================

def visualize_random_samples(image_dir, label_dir, num_samples=9, class_names=None):
    """
    Display random training images with bounding boxes
    """
    # Get all label files
    label_files = list(Path(label_dir).glob("*.txt"))
    
    # Select random files
    if len(label_files) > num_samples:
        selected = random.sample(label_files, num_samples)
    else:
        selected = label_files
    
    # Setup grid for display
    cols = 3
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() if num_samples > 1 else [axes]
    
    colors = plt.cm.tab20.colors  # Colors for different classes
    
    for idx, label_file in enumerate(selected):
        # Path to image
        img_path = Path(image_dir) / f"{label_file.stem}.jpg"
        if not img_path.exists():
            img_path = Path(image_dir) / f"{label_file.stem}.png"
        
        if not img_path.exists():
            axes[idx].text(0.5, 0.5, f"No image: {label_file.stem}", 
                          ha='center', va='center')
            axes[idx].axis('off')
            continue
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Read annotations
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Draw bounding boxes
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                # Fix: Convert float string (e.g., '21.0') to int
                class_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                
                # Convert from YOLO format to pixel coordinates
                x1 = int((xc - bw/2) * w)
                y1 = int((yc - bh/2) * h)
                x2 = int((xc + bw/2) * w)
                y2 = int((yc + bh/2) * h)
                
                # Draw rectangle
                color = colors[class_id % len(colors)]
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                                     linewidth=2, edgecolor=color, facecolor='none')
                axes[idx].add_patch(rect)
                
                # Class label
                label = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
                axes[idx].text(x1, y1 - 5, label, fontsize=8, 
                              color=color, weight='bold',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(f"{label_file.stem[:20]}", fontsize=10)
    
    # Hide unused axes
    for idx in range(len(selected), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('../outputs/sample_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f" Saved: ../outputs/sample_visualization.png")


# ============================================
# SIMPLE YOLO VISUALIZATION FUNCTIONS
# For Traffic Sign Detection Project
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import cv2


def load_results(exp_name, base_dir="../models/detect"):
    """Load training results CSV"""
    path = Path(base_dir) / exp_name / "results.csv"
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(f"Loaded {len(df)} epochs from {exp_name}")
    return df


def plot_losses(df):
    """Plot training losses"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['train/box_loss'], label='Box', linewidth=2)
    plt.plot(df['epoch'], df['train/cls_loss'], label='Cls', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', color='green', linewidth=2)
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='red', linewidth=2)
    plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision & Recall')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_map_progress(df):
    """Plot mAP50-95 progress with peak annotation"""
    plt.figure(figsize=(10, 5))
    
    best_epoch = df['metrics/mAP50-95(B)'].idxmax()
    best_value = df.loc[best_epoch, 'metrics/mAP50-95(B)']
    
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', color='blue', linewidth=2)
    plt.fill_between(df['epoch'], 0, df['metrics/mAP50-95(B)'], alpha=0.3)
    plt.axhline(y=best_value, color='green', linestyle='--', label=f'Best: {best_value:.4f}')
    plt.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('mAP50-95')
    plt.title(f'Best mAP50-95: {best_value:.4f} at epoch {best_epoch}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_epoch, best_value


def plot_confusion_matrix(exp_name, base_dir="../models/detect"):
    """Display confusion matrix"""
    path = Path(base_dir) / exp_name / "confusion_matrix.png"
    if path.exists():
        from IPython.display import Image
        display(Image(filename=str(path), width=800))
    else:
        print("Confusion matrix not found")



def print_summary(df):
    """Print key metrics summary"""
    best_epoch = df['metrics/mAP50-95(B)'].idxmax()
    best_map = df.loc[best_epoch, 'metrics/mAP50-95(B)']
    final_map = df['metrics/mAP50-95(B)'].iloc[-1]
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Best mAP50-95:  {best_map:.4f} (epoch {best_epoch})")
    print(f"Final mAP50-95: {final_map:.4f}")
    print(f"Best mAP50:     {df['metrics/mAP50(B)'].max():.4f}")
    print(f"Best Precision: {df['metrics/precision(B)'].max():.4f}")
    print(f"Best Recall:    {df['metrics/recall(B)'].max():.4f}")
    print("="*50)
    
    # Early stopping suggestion
    epochs_since_best = len(df) - best_epoch
    if epochs_since_best > 20:
        print(f"\n  {epochs_since_best} epochs wasted after peak!")
        print(f"   Try: patience={epochs_since_best + 5} or lower")
    else:
        print(f"\n Good! Only {epochs_since_best} epochs after peak")


def quick_analyze(exp_name):
    """Run all basic visualizations at once"""
    print(f"\n🔍 Analyzing: {exp_name}")
    print("="*50)
    
    df = load_results(exp_name)
    plot_losses(df)
    plot_map_progress(df)
    print_summary(df)
    
    # Try to show confusion matrix
    plot_confusion_matrix(exp_name)



