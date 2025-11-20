import re
import os
import matplotlib.pyplot as plt

# ============================================================
# 1. FUNCTION TO PARSE YOUR LOG FORMAT
# ============================================================
def parse_log(filepath):
    epochs = []
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    f1_scores = []

    # Match your log format exactly
    pattern = (
        r"Epoch\s+(\d+):\s+"
        r"TrainAcc=([0-9.]+)\s+\|\s+"
        r"ValAcc=([0-9.]+)\s+\|\s+"
        r"TrainLoss=([0-9.]+)\s+\|\s+"
        r"ValLoss=([0-9.]+)\s+\|\s+"
        r"F1=([0-9.]+)"
    )

    with open(filepath, "r") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                train_acc.append(float(match.group(2)))
                val_acc.append(float(match.group(3)))
                train_loss.append(float(match.group(4)))
                val_loss.append(float(match.group(5)))
                f1_scores.append(float(match.group(6)))

    return {
        "epochs": epochs,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "f1": f1_scores
    }

# ============================================================
# 2. REGISTER ALL YOUR LOGS HERE
# ============================================================
log_paths = {
    "DeepCNN": "logs/train_2025-11-10_10-19-45.log",
    "SimpleCNN": "logs/train_2025-11-09_14-54-16.log",
    "ViT_B16": "logs/train_2025-11-18_21-54-13.log",
    "OpenCLIP_ViT_B16": "logs/train_2025-11-19_09-05-14.log"
}

# ============================================================
# 3. LOAD ALL LOG DATA
# ============================================================
models = {}
for name, path in log_paths.items():
    try:
        models[name] = parse_log(path)
        print(f"Loaded log: {name}")
    except Exception as e:
        print(f"❌ Could not load {name}: {e}")

# ============================================================
# Ensure output directory exists
# ============================================================
os.makedirs("plots", exist_ok=True)

# ============================================================
# 4. PLOTTING FUNCTION — SAVES PNG, DOESN'T SHOW
# ============================================================
def plot_metric(metric_key, title, ylabel, filename):
    plt.figure(figsize=(10, 6))

    for name, data in models.items():
        if len(data["epochs"]) > 0:
            plt.plot(data["epochs"], data[metric_key], label=name, linewidth=2)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save image
    output_path = f"plots/{filename}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"📁 Saved: {output_path}")

# ============================================================
# 5. GENERATE ALL CURVES
# ============================================================

plot_metric("f1", "F1 Score per Epoch (All Models)", "F1 Score", "F1_AllModels")

plot_metric("train_loss", "Training Loss per Epoch (All Models)", "Loss", "TrainLoss_AllModels")

plot_metric("val_loss", "Validation Loss per Epoch (All Models)", "Loss", "ValLoss_AllModels")

plot_metric("train_acc", "Training Accuracy per Epoch (All Models)", "Accuracy", "TrainAcc_AllModels")

plot_metric("val_acc", "Validation Accuracy per Epoch (All Models)", "Accuracy", "ValAcc_AllModels")
