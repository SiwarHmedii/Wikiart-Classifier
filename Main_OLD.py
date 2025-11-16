# ==========================================================
# 🎨 WikiArt Classification - Model-Aware Training Pipeline
# ==========================================================
# Author: Siwar
# Description: Trains multiple CNNs with model-specific
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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ------------------------------
# 1️⃣ Reproducibility & Device
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {DEVICE}")

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
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # display live in terminal
        self.log.write(message)       # save in file
        self.log.flush()              # ensure immediate write

    def flush(self):
        # Needed for Python’s IO system to work properly
        self.terminal.flush()
        self.log.flush()

# Redirect stdout (print) to both terminal and log file
sys.stdout = Logger(log_path)

print(f"📝 Logging to: {log_path}\n")


# ==========================================================
# 3️⃣ TRANSFORM FUNCTION (Parametrized for Each Model)
# ==========================================================
def get_transforms(size, rotation=20, color_jitter=(0.3, 0.3, 0.2, 0.05)):
    """
    Return a tuple (train_tf, val_tf) for the given image size and augmentation params.
    """
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(rotation),
        transforms.ColorJitter(*color_jitter),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
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
        # class_to_idx is provided externally (consistent mapping)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label_str = self.labels[idx]
        label = self.class_to_idx[label_str]
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

# ==========================================================
# 6️⃣ OPTIONAL SAMPLING (FOR QUICK TESTING)
# ==========================================================
MAX_IMAGES_PER_CLASS = None  # set to an int for fast testing, or None for full dataset

if MAX_IMAGES_PER_CLASS is not None:
    print(f"⚡ Sampling up to {MAX_IMAGES_PER_CLASS} images per class for quick testing...")
    sampled_images, sampled_labels = [], []
    class_counter = defaultdict(int)
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images_shuf, all_labels_shuf = zip(*combined)
    for img, label in zip(all_images_shuf, all_labels_shuf):
        if class_counter[label] < MAX_IMAGES_PER_CLASS:
            sampled_images.append(img)
            sampled_labels.append(label)
            class_counter[label] += 1
    all_images, all_labels = sampled_images, sampled_labels
    print(f"✅ Using {len(all_images)} images across {len(set(all_labels))} classes.\n")
else:
    # ensure lists (train_test_split later expects sequences)
    all_images, all_labels = list(all_images), list(all_labels)

# ==========================================================
# 7️⃣ TRAIN/VAL/TEST SPLIT (stratified)
# ==========================================================
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.15, stratify=all_labels, random_state=SEED)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, stratify=train_labels, random_state=SEED)

# Create a consistent class -> idx mapping used by ALL datasets
class_names = sorted(list(set(all_labels)))
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
NUM_CLASSES = len(class_names)

print(f"📊 Data split summary:")
print(f"   ➤ Total images: {len(all_images)}")
print(f"   ➤ Classes: {NUM_CLASSES}")
print(f"   ➤ Train: {len(train_paths)}")
print(f"   ➤ Val:   {len(val_paths)}")
print(f"   ➤ Test:  {len(test_paths)}\n")

# ==========================================================
# 8️⃣ HANDLE CLASS IMBALANCE (Weighted Sampler)
# ==========================================================
# ==========================================================
# 8️⃣ HANDLE CLASS IMBALANCE (Weighted Sampler + Loss Weights)---->updated
# ==========================================================
cls_count = Counter(train_labels)

# Inverse-sqrt weighting
cls_weights = {cls: 1.0 / np.sqrt(count) for cls, count in cls_count.items()}

# Weighted sampler (balances mini-batches)
sample_weights = [cls_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Create tensor of weights in class index order for loss
loss_weights = torch.tensor(
    [cls_weights[c] for c in class_names], dtype=torch.float32
).to(DEVICE)


# ==========================================================
# 9️⃣ DATALOADER CREATION FUNCTION (accepts transforms)
# ==========================================================
def create_loaders(batch_size, train_tf, val_tf):
    """
    Build train/val/test loaders using the global class_to_idx mapping.
    num_workers=0 for Windows stability. drop_last=True for training to avoid BN with tiny final batch.
    this version has changed to be ran on WSL so now the num of workers can safely be increased to 8, sinec WSL handles multiprocessing better.
    8 workers setup is compatible with the machine used (16GB RAM, 4 cores) + NVIDIA GPU.
    
    """
    train_ds = WikiArtDataset(train_paths, train_labels, class_to_idx=class_to_idx, transform=train_tf)
    val_ds = WikiArtDataset(val_paths, val_labels, class_to_idx=class_to_idx, transform=val_tf)
    test_ds = WikiArtDataset(test_paths, test_labels, class_to_idx=class_to_idx, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=8, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)
    return train_loader, val_loader, test_loader


# ==========================================================
# 🔹 1️⃣ Simple CNN (size-adaptive)
# ==========================================================
class SimpleCNN(nn.Module):
    """
    Small CNN where the FC input is inferred dynamically from a dummy forward
    to avoid shape mismatches for different img_size values.
    """
    def __init__(self, num_classes, img_size=224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )

        # Dynamically compute flattened feature dimension for the classifier
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            n_features = self.features(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ==========================================================
# 🔹 2️⃣ Deep CNN (uses adaptive pooling so image size flexible)
# ==========================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.net(x)


class DeepCNN(nn.Module):
    """
    Uses AdaptiveAvgPool so fully connected input size is always 512.
    Safe for different input sizes.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = ConvBlock(3, 64)
        self.layer2 = ConvBlock(64, 128)
        self.layer3 = ConvBlock(128, 256)
        self.layer4 = ConvBlock(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ==========================================================
# 🔹 4️⃣ Pretrained Models (ResNet50 / EfficientNetV2)
# ==========================================================
'''def get_resnet50(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V1") #IMAGENET1K_V2
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model'''

from torchvision import models
import torch.nn as nn

def get_resnet50(num_classes):
    weights = models.ResNet50_Weights.IMAGENET1K_V2  # ✅ correct enum for V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_efficientnet(num_classes):
    # Use EfficientNetV2-S from torchvision
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1  # or None for random init
    model = models.efficientnet_v2_s(weights=weights)
    
    # Replace the classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model



# ==========================================================
# ✅ openclip_vitb16 / best#1
# ==========================================================
import open_clip
import timm


def get_openclip_vitb16(num_classes):
       # Load pretrained OpenCLIP ViT-B/16
    model = timm.create_model(
        'vit_base_patch16_clip_224.laion2b',  # pretrained on LAION-2B dataset
        pretrained=True,
        num_classes=num_classes
    )

    return model


#===========================================================
#Vision Transformer (ViT-B/16) pretrained on ImageNet-21K.
#=============================================================
def get_vit_base_in21k(num_classes):
    """
    Vision Transformer (ViT-B/16) pretrained on ImageNet-21K.
    Excellent generalization and color/style awareness.
    """
    model = timm.create_model(
        'vit_base_patch16_224_in21k',  # ViT-B/16 pretrained on ImageNet-21K
        pretrained=True,
        num_classes=num_classes        # replaces head automatically
    )
    return model


# ==========================================================
# 🔹 5️⃣ Training Function (shared)
# ==========================================================
def train_model(model, train_loader, val_loader, epochs, lr, name):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val = 0
    patience, counter = 8, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Training {name} Epoch {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0

        # Validation
        model.eval()
        val_correct, val_total, preds, labels_all = 0, 0, [], []
        total_val_loss = 0.0 
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_loss = criterion(outputs, labels) 
                preds_batch = outputs.argmax(1)
                val_correct += (preds_batch == labels).sum().item()
                val_total += labels.size(0)
                total_val_loss += val_loss.item() * imgs.size(0)
                preds.extend(preds_batch.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_f1 = f1_score(labels_all, preds, average='macro') if len(preds) > 0 else 0.0
        avg_val_loss = total_val_loss / val_total if val_total > 0 else 0.0   
        scheduler.step(val_acc)

        print( 
            f"[{name}] Epoch {epoch}: "
            f"TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f} | "
            f"TrainLoss={avg_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
            f"F1={val_f1:.3f}"
        )



        # Save the best model
        

        if val_f1 > best_val:
            best_val = val_f1
            counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{name}_best.pth")
            print(f"✅ Best {name} model saved (F1={best_val:.3f})!")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping (no F1 improvement).")
                break

    print(f"🎯 Best {name} ValAcc: {val_acc:.3f}|ValF1: {best_val:.3f}")
    return model


# ==========================================================
# 🔹 6️⃣ Evaluation Function
# ==========================================================
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
import numpy as np

def evaluate_model(model, loader, class_names):
    model.eval()
    correct, total, preds, labels_all = 0, 0, [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            p = outputs.argmax(1)
            correct += (p == labels).sum().item()
            total += labels.size(0)
            preds.extend(p.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    # Metrics
    acc = correct / total if total > 0 else 0.0
    error_rate = 1 - acc

    macro_f1 = f1_score(labels_all, preds, average='macro')
    weighted_f1 = f1_score(labels_all, preds, average='weighted')
    macro_precision = precision_score(labels_all, preds, average='macro', zero_division=0)
    weighted_precision = precision_score(labels_all, preds, average='weighted', zero_division=0)
    macro_recall = recall_score(labels_all, preds, average='macro', zero_division=0)
    weighted_recall = recall_score(labels_all, preds, average='weighted', zero_division=0)

    cm = confusion_matrix(labels_all, preds)
    report = classification_report(labels_all, preds, target_names=class_names, zero_division=0)

    # Display summary
    print("\n==============================")
    print("📊 MODEL EVALUATION SUMMARY")
    print("==============================")
    print(f"✅ Accuracy:       {acc:.4f}")
    print(f"❌ Error Rate:     {error_rate:.4f}")
    print(f"🎯 Macro F1:       {macro_f1:.4f}")
    print(f"📦 Weighted F1:    {weighted_f1:.4f}")
    print(f"✨ Macro Precision: {macro_precision:.4f}")
    print(f"✨ Weighted Prec.:  {weighted_precision:.4f}")
    print(f"📈 Macro Recall:    {macro_recall:.4f}")
    print(f"📈 Weighted Recall: {weighted_recall:.4f}")
    print(f"🧮 Confusion Matrix: {cm.shape}")
    print("==============================\n")
    print(report)

    return {
        "accuracy": acc,
        "error_rate": error_rate,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "weighted_precision": weighted_precision,
        "macro_recall": macro_recall,
        "weighted_recall": weighted_recall,
        "confusion_matrix": cm
    }


# ==========================================================
# 🔹 7️⃣ Model-Specific Configurations (edit as needed)
# ==========================================================
model_configs = {
   #"SimpleCNN": {"img_size": 128, "rotation": 15, "color_jitter": (0.2, 0.2, 0.2, 0.05), "batch": 128, "epochs": 50, "lr": 0.0005}, #30
  # "DeepCNN": {"img_size": 224, "rotation": 20, "color_jitter": (0.3, 0.3, 0.2, 0.05), "batch": 32, "epochs": 70, "lr": 5e-4}, #60
   #"ResNet50": {"img_size": 224, "rotation": 30, "color_jitter": (0.4, 0.4, 0.2, 0.1), "batch": 16, "epochs": 40, "lr": 2e-5}, #40
   #"EfficientNetV2": {"img_size": 224, "rotation": 20, "color_jitter": (0.3, 0.3, 0.2, 0.05), "batch": 16, "epochs": 50, "lr": 1e-4}, #50
  # "openclip_vitb16": {"img_size": 224, "rotation": 25, "color_jitter": (0.4, 0.4, 0.3, 0.1), "batch": 16, "epochs": 40, "lr": 1e-5}, 
   "vit_base_in21k": {"img_size": 224, "rotation": 25, "color_jitter": (0.4, 0.4, 0.3, 0.1), "batch": 16, "epochs": 40, "lr": 1e-5}, 
}

# ==========================================================
# 🔹 8️⃣ Model factory (map names to constructors)
# ==========================================================
def build_model_by_name(name, num_classes, img_size):
    """
    Return an instantiated model for the given name.
    Pass img_size where the constructor accepts it (SimpleCNN,etc).
    """
    if name == "SimpleCNN":
        return SimpleCNN(num_classes, img_size=img_size)
    elif name == "DeepCNN":
        return DeepCNN(num_classes)
    elif name == "ResNet50":
        return get_resnet50(num_classes)
    elif name == "EfficientNetV2":
        return get_efficientnet(num_classes)
    elif name == "openclip_vitb16":
        return get_openclip_vitb16(num_classes)
    elif name == "vit_base_in21k":
        return get_vit_base_in21k(num_classes)
        
    else:
        raise ValueError(f"Unknown model name: {name}")


# ==========================================================
# 🔹 9️⃣ Main model loop — only runs models defined in model_configs
# ==========================================================
if __name__ == "__main__":
    # iterate over configured models only
    for name, cfg in model_configs.items():
        print(f"\n==============================")
        print(f"🚀 Starting {name} | IMG={cfg['img_size']} | Batch={cfg['batch']} | LR={cfg['lr']} | Epochs={cfg['epochs']}")
        print(f"==============================\n")

        # Build model (pass img_size for those that need it)
        model = build_model_by_name(name, NUM_CLASSES, img_size=cfg["img_size"])

        # Create transforms for this model (model-specific augmentation)
        train_tf, val_tf = get_transforms(cfg["img_size"], cfg["rotation"], cfg["color_jitter"])

        # Create loaders using the transforms
        train_loader, val_loader, test_loader = create_loaders(cfg["batch"], train_tf, val_tf)

        # Train
        model = train_model(model, train_loader, val_loader, cfg["epochs"], cfg["lr"], name)

        # Load best checkpoint and evaluate on test set
        ckpt_path = CHECKPOINT_DIR / f"{name}_best.pth"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            evaluate_model(model, test_loader, class_names)
        else:
            print(f"⚠️ No checkpoint found for {name}, skipping final evaluation.")

    print("\n🏁 All configured models processed.")
