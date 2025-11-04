# ==========================================================
# 🌐 WikiArt API - Fancy Image Classification Web App
# ==========================================================
# Author: Siwar
# Description: Flask-based API + Web UI for WikiArt Classification
#source .venv/bin/activate
#http://localhost:7860
# ==========================================================

import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, render_template, jsonify
from pathlib import Path


# ----------------------------
# 🔧 CONFIGURATION
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("./checkpoints") / "ResNet50_best.pth"
NUM_CLASSES = 27  # update with your dataset
CLASS_NAMES = sorted(os.listdir(Path.home() / "wikiart_project/wikiart"))

# ----------------------------
# 🧠 LOAD MODEL
# ----------------------------
def load_model():
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    print("✅ Model loaded successfully.")
    return model

model = load_model()

# ----------------------------
# 🖼️ TRANSFORM FOR INFERENCE
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----------------------------
# 🔮 PREDICTION FUNCTION
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

    # Save uploaded image temporarily
    filepath = os.path.join("static/uploads", file.filename)
    os.makedirs("static/uploads", exist_ok=True)
    file.save(filepath)

    # Run prediction
    preds = predict_image(filepath)
    return jsonify({
        "image_path": filepath,
        "predictions": [{"class": c, "prob": float(p)} for c, p in preds]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
