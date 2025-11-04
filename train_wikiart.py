# ==========================================================
# 🧠 WIKIART TRAINING SCRIPT (WSL2 optimized)
# ==========================================================
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# ==========================================================
# 0️⃣ BASIC SETUP
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# Paths — store checkpoints/logs in Linux home (faster)
BASE_DIR = Path.home() / "wikiart_project"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
DATASET_DIR = BASE_DIR / "wikiart"  # or change to Linux copy

for d in [CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
print(f"📂 Dataset directory: {DATASET_DIR}")
print(f"💾 Checkpoints will be saved in: {CHECKPOINT_DIR}")
print(f"🧾 Logs in: {LOG_DIR}")

import sys
from datetime import datetime

log_file = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
sys.stdout = open(log_file, "a", buffering=1)
sys.stderr = sys.stdout

print(f"📝 Logging all console output to: {log_file}\n")


# ==========================================================
# 1️⃣ LOGGING SETUP
# ==========================================================
log_file = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
sys.stdout = open(log_file, "a", buffering=1, encoding="utf-8")
sys.stderr = sys.stdout
print(f"📄 Logging all outputs to: {log_file}\n")

# ==========================================================
# 2️⃣ TRANSFORMS
# ==========================================================
def get_transforms(img_size, rotation, color_jitter):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(rotation),
        transforms.ColorJitter(*color_jitter),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

# ==========================================================
# 3️⃣ DATASET + LOADERS
# ==========================================================
def create_loaders(batch_size, train_tf, val_tf):
    from torchvision.datasets import ImageFolder

    train_path = DATASET_DIR / "train"
    val_path = DATASET_DIR / "val"
    test_path = DATASET_DIR / "test"

    train_ds = ImageFolder(train_path, transform=train_tf)
    val_ds = ImageFolder(val_path, transform=val_tf)
    test_ds = ImageFolder(test_path, transform=val_tf)

    train_labels = [label for _, label in train_ds.samples]
    cls_count = Counter(train_labels)
    cls_weights = {cls: 1.0 / np.sqrt(count) for cls, count in cls_count.items()}
    sample_weights = [cls_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, len(train_ds.classes)

# ==========================================================
# 4️⃣ MODEL DEFINITIONS
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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // reduction, 1), nn.ReLU(),
            nn.Conv2d(in_ch // reduction, in_ch, 1), nn.Sigmoid()
        )
    def forward(self, x): return x * self.fc(x)


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
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
# 5️⃣ TRAINING FUNCTION
# ==========================================================
def train_model(model, train_loader, val_loader, epochs, lr, name):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_val, patience, counter = 0, 10, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"{name} Epoch {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
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

        val_acc = val_correct / val_total
        val_f1 = f1_score(labels_all, preds, average='macro')
        scheduler.step(val_acc)

        print(f"[{name}] Epoch {epoch}: TrainAcc={train_acc:.3f} | ValAcc={val_acc:.3f} | F1={val_f1:.3f}")

        if val_acc > best_val:
            best_val, counter = val_acc, 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"{name}_best.pth")
            print(f"✅ Best {name} model saved!")
        else:
            counter += 1
            if counter >= patience:
                print("⏹️ Early stopping.")
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

    acc = correct / total
    cm = confusion_matrix(labels_all, preds)
    report = classification_report(labels_all, preds, target_names=class_names)
    print(f"✅ Test Accuracy: {acc:.3f}")
    print(report)
    return acc, cm


# ==========================================================
# 6️⃣ CONFIG + MAIN LOOP
# ==========================================================
model_configs = {
    "SimpleCNN": {"img_size": 128, "rotation": 15, "color_jitter": (0.2, 0.2, 0.1, 0.05), "batch": 128, "epochs": 1, "lr": 5e-4},
   # "DeepCNN": {"img_size": 224, "rotation": 20, "color_jitter": (0.3, 0.3, 0.2, 0.05), "batch": 32, "epochs": 3, "lr": 5e-4},
    #"DeepCNN_v2": {"img_size": 224, "rotation": 25, "color_jitter": (0.4, 0.4, 0.3, 0.1), "batch": 32, "epochs": 3, "lr": 3e-4},
   # "ResNet50": {"img_size": 256, "rotation": 30, "color_jitter": (0.4, 0.4, 0.2, 0.1), "batch": 16, "epochs": 3, "lr": 2e-5},
    #"EfficientNetB0": {"img_size": 224, "rotation": 20, "color_jitter": (0.3, 0.3, 0.2, 0.05), "batch": 16, "epochs": 3, "lr": 1e-4},
}

def build_model_by_name(name, num_classes, img_size):
    if name == "SimpleCNN":
        return SimpleCNN(num_classes, img_size)
    elif name == "DeepCNN":
        return DeepCNN(num_classes)
    elif name == "DeepCNN_v2":
        return DeepCNN_v2(num_classes, img_size)
    elif name == "ResNet50":
        return get_resnet50(num_classes)
    elif name == "EfficientNetB0":
        return get_efficientnet(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")

if __name__ == "__main__":
    for name, cfg in model_configs.items():
        print(f"\n==============================")
        print(f"🚀 Starting {name} | IMG={cfg['img_size']} | Batch={cfg['batch']} | LR={cfg['lr']} | Epochs={cfg['epochs']}")
        print(f"==============================\n")

        train_tf, val_tf = get_transforms(cfg["img_size"], cfg["rotation"], cfg["color_jitter"])
        train_loader, val_loader, test_loader, NUM_CLASSES = create_loaders(cfg["batch"], train_tf, val_tf)
        class_names = [f"Class {i}" for i in range(NUM_CLASSES)]
        model = build_model_by_name(name, NUM_CLASSES, cfg["img_size"])

        model = train_model(model, train_loader, val_loader, cfg["epochs"], cfg["lr"], name)
        ckpt = CHECKPOINT_DIR / f"{name}_best.pth"
        if ckpt.exists():
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            evaluate_model(model, test_loader, class_names)
        else:
            print(f"⚠️ No checkpoint found for {name}, skipping evaluation.")

    print("\n🏁 All models processed.")
