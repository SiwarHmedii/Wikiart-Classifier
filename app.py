# ==========================================================
# 🌐 WikiArt API - Fancy Image Classification Web App
# ==========================================================
# Authors: Siwar Hmedi / Mariem Arfaoui/ Anwar Hriz/ Zied Touahri / Mahdi Younsi/ Nour Bayoudh
# Description: Flask-based API + Web UI for WikiArt Classification
#source .venv/bin/activate
#http://localhost:7860# ==========================================================
# 🌐 WikiArt API - Fancy Image Classification Web App
# ==========================================================
# Author: Siwar
# Description: Flask-based API + Web UI for WikiArt Classification
# ==========================================================

import os
import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, render_template, jsonify
from pathlib import Path

import timm                   # <-- You MUST import timm now
from torchvision import transforms
# ----------------------------
# 🔧 CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("./checkpoints") / "openclip_vitb16_best.pth"


NUM_CLASSES = 27

# Load only valid class names (ignore junk folders)
all_items = sorted(os.listdir(Path.home() / "wikiart_project/wikiart"))
CLASS_NAMES = [d for d in all_items if not d.startswith(".")][:NUM_CLASSES]


# ----------------------------
# 🧠 LOAD MODEL  (OPENCLIP ViT-B/16)
# ----------------------------
def load_model():
    # Create the OpenCLIP ViT-B/16 model
    model = timm.create_model(
        "vit_base_patch16_clip_224",
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    # Load the checkpoint
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)

    model.eval().to(DEVICE)
    print("✅ OpenCLIP ViT-B/16 Model loaded successfully.")
    return model


model = load_model()


# ----------------------------
# 🖼️ TRANSFORM FOR INFERENCE
# (OpenCLIP ViT-B/16 uses CLIP normalization stats)
# ----------------------------
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.Resize(int(224 * 1.14)),  # CLIP resize (~256)
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(clip_mean, clip_std)
])


# ----------------------------
# 🔮 PREDICTION FUNCTION (UNCHANGED, ONLY MODEL INPUT FIXED)
# ----------------------------
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)

        top_probs, top_idxs = probs.topk(3, dim=1)

        top_probs = top_probs.cpu().numpy()[0]
        top_idxs = top_idxs.cpu().numpy()[0]

        top_classes = [CLASS_NAMES[i] for i in top_idxs]

    return list(zip(top_classes, top_probs))


# ==========================================================
# 🌍 FLASK APP SETUP
# ==========================================================
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join("static/uploads", file.filename)
    os.makedirs("static/uploads", exist_ok=True)
    file.save(filepath)

    preds = predict_image(filepath)

    return jsonify({
        "image_path": filepath,
        "predictions": [{"class": c, "prob": float(p)} for c, p in preds]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
