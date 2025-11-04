# ==========================================================
# 🎨 WikiArt Classification - Model-Aware Training Pipeline (Linux Version)
# ==========================================================
# Author: Siwar
# Description: Trains multiple CNNs with model-specific
#              augmentations, input sizes, and hyperparameters.
#              Optimized for Linux (uses multiple workers).
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
# Update these to match your Linux paths
SOURCE_DIR = Path("/home/siwar//wikiart_project/wikiart")  # Dataset folder
CHECKPOINT_DIR = Path("/home/siwar/wikiart_checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# Linux supports multiprocessing for dataloaders
NUM_WORKERS = max(2, os.cpu_count() // 2)

# ==========================================================
# 3️⃣ TRANSFORM FUNCTION
# ==========================================================
def get_transforms(size, rotation=20, color_jitter=(0.3, 0.3, 0.2, 0.05)):
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
# 4️⃣ CUSTOM DATASET CLASS
# ==========================================================
class WikiArtDataset(Dataset):
    def __init__(self, image_paths, labels, class_to_idx, transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.class_to_idx[self.labels[idx]]
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
# 6️⃣ OPTIONAL SAMPLING
# ==========================================================
MAX_IMAGES_PER_CLASS = None  # e.g., 200 for testing

if MAX_IMAGES_PER_CLASS is not None:
    print(f"⚡ Sampling up to {MAX_IMAGES_PER_CLASS} images per class...")
    sampled_images, sampled_labels = [], []
    class_counter = defaultdict(int)
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    for img, label in combined:
        if class_counter[label] < MAX_IMAGES_PER_CLASS:
            sampled_images.append(img)
            sampled_labels.append(label)
            class_counter[label] += 1
    all_images, all_labels = sampled_images, sampled_labels
    print(f"✅ Using {len(all_images)} images across {len(set(all_labels))} classes.\n")

# ==========================================================
# 7️⃣ TRAIN/VAL/TEST SPLIT
# ==========================================================
train_paths, test_paths, train_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.15, stratify=all_labels, random_state=SEED)
train_paths, val_paths, train_labels, val_labels = train_test_split(
    train_paths, train_labels, test_size=0.15, stratify=train_labels, random_state=SEED)

class_names = sorted(list(set(all_labels)))
class_to_idx = {cls: i for i, cls in enumerate(class_names)}
NUM_CLASSES = len(class_names)

print(f"📊 Data split summary:")
print(f"   ➤ Total: {len(all_images)}")
print(f"   ➤ Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
print(f"   ➤ Classes: {NUM_CLASSES}\n")

# ==========================================================
# 8️⃣ HANDLE CLASS IMBALANCE
# ==========================================================
cls_count = Counter(train_labels)
cls_weights = {cls: 1.0 / np.sqrt(count) for cls, count in cls_count.items()}
sample_weights = [cls_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
loss_weights = torch.tensor([cls_weights[c] for c in class_names], dtype=torch.float32).to(DEVICE)


# ==========================================================
# 9️⃣ DATALOADER CREATION
# ==========================================================
def create_loaders(batch_size, train_tf, val_tf):
    train_ds = WikiArtDataset(train_paths, train_labels, class_to_idx, transform=train_tf)
    val_ds = WikiArtDataset(val_paths, val_labels, class_to_idx, transform=val_tf)
    test_ds = WikiArtDataset(test_paths, test_labels, class_to_idx, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader, test_loader


# ==========================================================
# 🔹 MODELS
# ==========================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes, img_size=224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            n_features = self.features(dummy).view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(self.features(x), 1)
        return self.classifier(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x): return self.net(x)


class DeepCNN(nn.Module):
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
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // reduction, 1), nn.ReLU(),
            nn.Conv2d(in_ch // reduction, in_ch, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.use_se = use_se
        if use_se: self.se = SEBlock(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.use_se: out = self.se(out)
        out += self.shortcut(x)
        return self.relu(out)


class DeepCNN_v2(nn.Module):
    def __init__(self, num_classes, img_size=224, use_se=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.stage1 = ResidualBlock(64, 128, use_se)
        self.stage2 = ResidualBlock(128, 256, use_se)
        self.stage3 = ResidualBlock(256, 512, use_se)
        self.stage4 = ResidualBlock(512, 512, use_se)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            x = stage(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def get_resnet50(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_efficientnet(num_classes):
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


# ==========================================================
# 🔹 TRAINING & EVAL FUNCTIONS
# ==========================================================
def train_model(model, train_loader, val_loader, epochs, lr, name):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_val = 0
    patience, counter = 12, 0

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

        model.eval()
        val_correct, val_total, preds, labels_all = 0, 0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                preds_batch = outputs.argmax(1)
                val_correct += (preds_batch == labels).sum().item()
                val_total += labels.size(0)
                preds.extend(preds_batch.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_f1 = f1_score(labels_all, preds, average='macro') if len(preds) > 0 else 0.0
        scheduler.step(val_acc)

        print(f"[{name}] Epoch {epoch}: TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f} | F1={val_f1:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{name}_best.pth")
            print(f"✅ Saved best {name} model.")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping triggered.")
                break

    print(f"🎯 Best {name} ValAcc: {best_val:.3f}")
    return model


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
    acc = correct / total if total > 0 else 0.0
    cm = confusion_matrix(labels_all, preds) if len(preds) > 0 else None
    report = classification_report(labels_all, preds, target_names=class_names) if len(preds) > 0 else ""
    print(f"✅ Test Accuracy: {acc:.3f}")
    if report:
        print(report)
    return acc, cm


# ==========================================================
# 🔹 MODEL CONFIGS
# ==========================================================
model_configs = {
    "SimpleCNN": {"img_size": 128, "rotation": 15, "color_jitter": (0.2, 0.2, 0.1, 0.05), "batch": 128, "epochs": 3, "lr": 5e-4},
    "DeepCNN": {"img_size": 224, "rotation": 20, "color_jitter": (0.3, 0.3, 0.2, 0.05), "batch": 32, "epochs": 3, "lr": 5e-4},
    "DeepCNN_v2": {"img_size": 224, "rotation": 25, "color_jitter": (0.4, 0.4, 0.3, 0.1), "batch": 32, "epochs": 3, "lr": 3e-4},
    "ResNet50": {"img_size": 256, "rotation": 30, "color_jitter": (0.4, 0.4, 0.2, 0.1), "batch": 16, "epochs": 3, "lr": 2e-5},
    "EfficientNetB0": {"img_size": 224, "rotation": 20, "color_jitter": (0.3, 0.3, 0.2, 0.05), "batch": 16, "epochs": 3, "lr": 1e-4},
}


# ==========================================================
# 🔹 MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    for name, cfg in model_configs.items():
        print(f"\n==============================")
        print(f"🚀 Starting {name} | IMG={cfg['img_size']} | Batch={cfg['batch']} | LR={cfg['lr']} | Epochs={cfg['epochs']}")
        print(f"==============================\n")

        model = {
            "SimpleCNN": lambda: SimpleCNN(NUM_CLASSES, img_size=cfg["img_size"]),
            "DeepCNN": lambda: DeepCNN(NUM_CLASSES),
            "DeepCNN_v2": lambda: DeepCNN_v2(NUM_CLASSES, img_size=cfg["img_size"]),
            "ResNet50": lambda: get_resnet50(NUM_CLASSES),
            "EfficientNetB0": lambda: get_efficientnet(NUM_CLASSES)
        }[name]()

        train_tf, val_tf = get_transforms(cfg["img_size"], cfg["rotation"], cfg["color_jitter"])
        train_loader, val_loader, test_loader = create_loaders(cfg["batch"], train_tf, val_tf)

        model = train_model(model, train_loader, val_loader, cfg["epochs"], cfg["lr"], name)
        ckpt_path = CHECKPOINT_DIR / f"{name}_best.pth"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            evaluate_model(model, test_loader, class_names)
        else:
            print(f"⚠️ No checkpoint found for {name}. Skipping evaluation.")

    print("\n🏁 All configured models processed.")
