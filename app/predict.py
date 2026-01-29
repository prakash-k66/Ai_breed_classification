import json
import torch
import torch.nn as nn
from fastapi import APIRouter, UploadFile, File
from torchvision import transforms, models
from PIL import Image
import io

router = APIRouter()

# --------------------------------------------------
# Device
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Load class names
# --------------------------------------------------
with open("artifacts/class_names.json", "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(
    torch.load("artifacts/breed_model.pth", map_location=device)
)

model.to(device)
model.eval()

# --------------------------------------------------
# Image transform (same as training)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Predict API (TOP-3)
# --------------------------------------------------
@router.post("/predict")
async def predict_breed(image: UploadFile = File(...)):
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=3)

    predictions = []
    for prob, idx in zip(top_probs, top_idxs):
        predictions.append({
            "breed": class_names[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    top_confidence = predictions[0]["confidence"]

    if top_confidence >= 80:
        status = "CONFIDENT_PREDICTION"
    elif top_confidence >= 60:
        status = "NEEDS_HUMAN_CONFIRMATION"
    else:
        status = "UNKNOWN_BREED"

    return {
        "top_predictions": predictions,
        "status": status
    }
