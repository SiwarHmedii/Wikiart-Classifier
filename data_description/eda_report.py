# eda_report.py
import os
import random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

# ---------------------------
# CONFIG
# ---------------------------
SOURCE_DIR = Path.home() / "wikiart_project/wikiart"
REPORT_DIR = Path("./data_description")
REPORT_DIR.mkdir(exist_ok=True)

SAMPLE_GRID_SIZE = (4, 4)  # rows x cols
MAX_IMAGES_PER_CLASS = None  # set to int for testing smaller subset

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def load_images_info(source_dir):
    """Scan dataset, return dict with class -> list of image paths"""
    data = {}
    for cls_dir in source_dir.iterdir():
        if cls_dir.is_dir():
            imgs = [p for p in cls_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
            data[cls_dir.name] = imgs
    return data

def plot_sample_grid(image_paths, title, save_path):
    n = min(len(image_paths), SAMPLE_GRID_SIZE[0]*SAMPLE_GRID_SIZE[1])
    sample_imgs = random.sample(image_paths, n)
    
    fig, axes = plt.subplots(SAMPLE_GRID_SIZE[0], SAMPLE_GRID_SIZE[1], figsize=(12,12))
    for ax, img_path in zip(axes.flatten(), sample_imgs):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def detect_outliers(arr):
    """IQR-based outlier detection"""
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    outliers = (arr < lower) | (arr > upper)
    return outliers, lower, upper

def save_text_report(lines, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def plot_class_distribution(labels, title, save_path=None):
    counts = Counter(labels)
    classes, values = zip(*sorted(counts.items()))
    plt.figure(figsize=(12,5))
    sns.barplot(x=list(classes), y=list(values), color="skyblue")
    plt.xticks(rotation=90)
    plt.ylabel("Number of Images")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.close()
    return counts

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    report_lines = []
    
    # Load images
    data = load_images_info(SOURCE_DIR)
    all_counts = {cls: len(imgs) for cls, imgs in data.items()}
    total_images = sum(all_counts.values())
    num_classes = 27
    
    report_lines.append("========== DATASET SUMMARY ==========")
    report_lines.append(f"Total classes: {num_classes}")
    report_lines.append(f"Total images: {total_images}")
    
    counts_series = pd.Series(all_counts)
    report_lines.append("Class counts:")
    report_lines.append(str(counts_series))
    
    # Imbalance warning
    if counts_series.min() / counts_series.max() < 0.5:
        report_lines.append("\n⚠️ WARNING: Dataset is imbalanced! Consider weighted sampling.")
    else:
        report_lines.append("\nClass distribution is relatively balanced.")
    
    # Overall histogram
    plot_class_distribution(list(all_labels for all_labels in counts_series.index),
                            "Overall Class Distribution",
                            save_path=REPORT_DIR / "overall_class_distribution.png")
    
    # -----------------------
    # Train/Val/Test split
    # -----------------------
    all_images = [str(img) for imgs in data.values() for img in imgs]
    all_labels = [cls for cls, imgs in data.items() for _ in imgs]

    # Save counts before sampling
    plot_class_distribution(all_labels, "Class Distribution Before Sampler",
                            save_path=REPORT_DIR / "class_distribution_before_sampler.png")

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=0.15, stratify=all_labels, random_state=123)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.15, stratify=train_labels, random_state=123)

    # Create WeightedRandomSampler for training
    cls_count = Counter(train_labels)
    cls_weights = {cls: 1.0 / np.sqrt(count) for cls, count in cls_count.items()}
    sample_weights = [cls_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(list(zip(train_paths, train_labels)), sampler=sampler, batch_size=16)

    # Histogram after sampler
    sampled_labels = []
    for batch in train_loader:
        _, labels_batch = batch
        sampled_labels.extend(labels_batch)
    plot_class_distribution(sampled_labels, "Class Distribution After Sampler",
                            save_path=REPORT_DIR / "class_distribution_after_sampler.png")
    
    # Histogram per split
    def plot_split_hist(labels, split_name):
        counter = plot_class_distribution(labels, f"{split_name} Split Distribution",
                                          save_path=REPORT_DIR / f"{split_name.lower()}_distribution.png")
        return counter

    train_counter = plot_split_hist(train_labels, "Training")
    val_counter = plot_split_hist(val_labels, "Validation")
    test_counter = plot_split_hist(test_labels, "Test")

    # -----------------------
    # Image size stats & outliers
    # -----------------------
    widths, heights = [], []
    for img_path in all_images:
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    widths, heights = np.array(widths), np.array(heights)
    
    w_outliers, w_low, w_high = detect_outliers(widths)
    h_outliers, h_low, h_high = detect_outliers(heights)
    
    report_lines.append("\nImage size statistics (WxH):")
    report_lines.append(f"Width: mean={widths.mean():.1f}, min={widths.min()}, max={widths.max()}, outliers={w_outliers.sum()}")
    report_lines.append(f"Height: mean={heights.mean():.1f}, min={heights.min()}, max={heights.max()}, outliers={h_outliers.sum()}")
    
    # Boxplot
    plt.figure(figsize=(8,6))
    sns.boxplot(data=[widths, heights])
    plt.xticks([0,1], ["Width","Height"])
    plt.title("Image Dimension Outliers")
    plt.savefig(REPORT_DIR / "image_size_boxplot.png")
    plt.close()
    
    # Histogram
    plt.figure(figsize=(10,4))
    plt.hist(widths, bins=30, alpha=0.6, label="Width")
    plt.hist(heights, bins=30, alpha=0.6, label="Height")
    plt.axvline(w_low, color="r", linestyle="--")
    plt.axvline(w_high, color="r", linestyle="--")
    plt.axvline(h_low, color="g", linestyle="--")
    plt.axvline(h_high, color="g", linestyle="--")
    plt.legend()
    plt.title("Image Size Distribution & Outliers")
    plt.savefig(REPORT_DIR / "image_size_hist.png")
    plt.close()
    
    # Sample grid
    plot_sample_grid(all_images, "Random Samples Before Transformation", REPORT_DIR / "sample_grid.png")
    
    # -----------------------
    # Automated conclusions
    # -----------------------
    report_lines.append("\n========== AUTOMATED CONCLUSIONS ==========")
    
    # Class balance
    if counts_series.min() / counts_series.max() < 0.5:
        report_lines.append("- Classes are imbalanced. Consider weighted sampling or augmentation.")
    else:
        report_lines.append("- Class distribution is relatively balanced.")
    
    # After sampler
    report_lines.append("- WeightedRandomSampler applied: train batches now balanced (see plot).")
    
    # Outliers
    if w_outliers.sum() > 0 or h_outliers.sum() > 0:
        report_lines.append(f"- Detected image size outliers. Width threshold: [{w_low:.0f},{w_high:.0f}], Height threshold: [{h_low:.0f},{h_high:.0f}]")
    else:
        report_lines.append("- Image sizes are consistent; no significant outliers detected.")
    
    # Split balance
    for split_name, counter in [("Train", train_counter), ("Validation", val_counter), ("Test", test_counter)]:
        counts = np.array(list(counter.values()))
        if counts.min()/counts.max() < 0.5:
            report_lines.append(f"- ⚠️ {split_name} split shows class imbalance.")
        else:
            report_lines.append(f"- {split_name} split is relatively balanced.")
    
    report_lines.append(f"- Total classes: {num_classes}, total images: {total_images}")
    report_lines.append(f"- Sample grid saved to {REPORT_DIR / 'sample_grid.png'}")
    
    # Save textual report
    save_text_report(report_lines, REPORT_DIR / "eda_report.txt")
    print("✅ EDA report completed. Check plots and eda_report.txt in ./data_description/")
