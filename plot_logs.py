import re 
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

log_path = Path("logs/train_2025-11-18_21-54-13.log")

# Create output folder
out_dir = Path("data_evaluation/")
out_dir.mkdir(exist_ok=True)

# Regex to extract training lines
epoch_regex = re.compile(
    r"\[(.*?)\] Epoch (\d+): TrainAcc=([\d.]+) \| ValAcc=([\d.]+) \| "
    r"TrainLoss=([\d.]+) \| ValLoss=([\d.]+) \| F1=([\d.]+)"
)

# Storage
metrics = defaultdict(lambda: defaultdict(list))

with open(log_path, "r") as f:
    for line in f:
        match = epoch_regex.search(line)
        if match:
            model, epoch, tr_acc, val_acc, tr_loss, val_loss, f1 = match.groups()
            epoch = int(epoch)

            metrics[model]["epoch"].append(epoch)
            metrics[model]["train_acc"].append(float(tr_acc))
            metrics[model]["val_acc"].append(float(val_acc))
            metrics[model]["train_loss"].append(float(tr_loss))
            metrics[model]["val_loss"].append(float(val_loss))
            metrics[model]["f1"].append(float(f1))

# Plot for each model
for model, m in metrics.items():
    print(f"Plotting curves for: {model}")

    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(m["epoch"], m["train_acc"], label="Train Accuracy")
    plt.plot(m["epoch"], m["val_acc"], label="Val Accuracy")
    plt.title(f"{model} - Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / f"{model}_accuracy.png", dpi=150)
    plt.close()

    # Loss
    plt.figure(figsize=(8,5))
    plt.plot(m["epoch"], m["train_loss"], label="Train Loss")
    plt.plot(m["epoch"], m["val_loss"], label="Val Loss")
    plt.title(f"{model} - Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / f"{model}_loss.png", dpi=150)
    plt.close()

    # F1
    plt.figure(figsize=(8,5))
    plt.plot(m["epoch"], m["f1"], label="Val F1 Score", color="green")
    plt.title(f"{model} - F1 Score Curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.grid(True)
    plt.savefig(out_dir / f"{model}_f1.png", dpi=150)
    plt.close()

print("\n✅ All plots saved in: data_evaluation/")
