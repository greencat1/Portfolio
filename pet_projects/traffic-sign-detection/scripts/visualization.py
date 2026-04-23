import cv2
import random
import matplotlib.pyplot as plt
from pathlib import Path

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