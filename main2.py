# ==========================================================
# 🎨 WikiArt Classification - Model-Aware Training Pipeline
# ==========================================================
# # Description: Trains multiple CNNs with model-specific
#              augmentations, input sizes and hyperparameters.
# - Uses consistent class -> index mapping across datasets
# - SimpleCNN is dynamic (infers flattened size)
# - Safe defaults for Windows: num_workers=0
# ==========================================================

import os
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

from torchvision import models
import torch.nn as nn


import torch
# Clear cache only if CUDA available
if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.io import read_image
import torchvision.transforms.v2 as T
from PIL import Image


import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
# ------------------------------
# 1️⃣ Reproducibility & Device
# ------------------------------
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")




# Print GPU details only if available (defensive)
if DEVICE.type == 'cuda':
    try:
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM Total:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    except Exception as e:
        print("GPU info unavailable:", e)

# ------------------------------
# 2️⃣ Paths & Constants
# ------------------------------
SOURCE_DIR =  Path.home() / "wikiart_project/wikiart"   # dataset folder
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


#-------------------------------------------
# Save my LOGS
#-------------------------------------------


import sys
import datetime

# Create logs folder if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate timestamped log filename
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"logs/train_{timestamp}.log"

# Define a custom class that writes to both terminal and file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout  # original stdout
        self.log = open(filename, "a", encoding="utf-8") # log file in append mode

    def write(self, message): 
        self.terminal.write(message)  # display live in terminal
        self.log.write(message)       # save in file
        self.log.flush()              # ensure immediate write

    def flush(self):
        # Needed for Python’s IO system to work properly
        self.terminal.flush() # flush terminal
        self.log.flush()    # flush log file

# Redirect stdout (print) to both terminal and log file
sys.stdout = Logger(log_path)

print(f"📝 Logging to: {log_path}\n")

#=========================================================
# 3️⃣ TRANSFORM FUNCTION (Parametrized)
#==========================================================
def get_transforms(model_name, size, rotation, color_jitter):
    """Return (train_tf, val_tf). Standard pipeline with config parameters.
    """
    # Standard pipeline
    train_tf = T.Compose([
        T.Resize((size, size)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(rotation),
        T.ColorJitter(*color_jitter),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transforms: simple resize
    val_tf = T.Compose([
        T.Resize((size, size)),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_tf, val_tf




# ==========================================================
# 4️⃣ CUSTOM DATASET CLASS (uses global class_to_idx mapping)
# ==========================================================

class WikiArtDataset(Dataset):
    """
    Lazily loads images from disk and maps labels using a provided
    class_to_idx mapping to ensure consistent indices across splits.
    """
    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

        # mapping must be passed from outside
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # FAST C++ JPEG loader (much faster than Pillow)
        img = read_image(self.image_paths[idx])  # returns tensor [C,H,W] uint8

        # convert to float for transforms expecting [0, 1]
        img = img.float() / 255.0

        # map label
        label_str = self.labels[idx]
        label = self.class_to_idx[label_str]

        # apply transforms
        if self.transform:
            img = self.transform(img)

        return img, label

# ==========================================================
# 5️⃣ LOAD ALL IMAGES AND LABELS
# ==========================================================
all_images, all_labels = [], []
for cls in sorted(os.listdir(SOURCE_DIR)):
    cls_path = SOURCE_DIR / cls
    if not cls_path.is_dir():
        continue
    for img in cls_path.glob("*.*"):
        if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            all_images.append(str(img))
            all_labels.append(cls)

print(f"\n🖼️ Loaded {len(all_images)} total images across {len(set(all_labels))} classes.\n")




