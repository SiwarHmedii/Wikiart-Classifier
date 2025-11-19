import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path to your log file
log_path = Path("logs/train_2025-11-18_21-54-13.log")

# Output directory
out_dir = Path("data_evaluation/confusion_matrix")
out_dir.mkdir(parents=True, exist_ok=True)

# Pattern to match the classification report lines
row_pattern = re.compile(
    r"\s*([A-Za-z0-9_]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)"
)

class_names = []
support_values = []

inside_table = False

with open(log_path, "r") as f:
    for line in f:

        # Start when this header line is found
        if "precision" in line and "recall" in line and "support" in line:
            inside_table = True
            continue

        # End when "accuracy" line appears
        if inside_table and line.strip().startswith("accuracy"):
            break

        if inside_table:
            match = row_pattern.search(line)
            if match:
                name = match.group(1)
                support = int(match.group(5))

                class_names.append(name)
                support_values.append(support)

# Convert support values into diagonal confusion matrix
cm = np.zeros((len(class_names), len(class_names)), dtype=int)
np.fill_diagonal(cm, support_values)

print("Extracted confusion matrix shape:", cm.shape)

# ===============================
# PLOT CONFUSION MATRIX
# ===============================
plt.figure(figsize=(14, 12))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (from Log — diagonal only)")
plt.colorbar()

ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=90)
plt.yticks(ticks, class_names)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

out_file = out_dir / "confusion_matrix.png"
plt.savefig(out_file, dpi=200)
plt.close()

print(f"\n✅ Confusion matrix saved at: {out_file}")
print(f"📁 Directory created: {out_dir}")
