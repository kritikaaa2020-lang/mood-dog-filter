
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from models.emotion_model import create_emotion_model


DEVICE = torch.device("cpu")

EMOTION_CLASSES = ["Happy", "Sad", "Surprise"]

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_jaffe_predictor(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained JAFFE model not found: {model_path}"
        )

    model = create_emotion_model()

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    # Load the best saved ResNet18 weights.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    def predict(frame):
        if frame is None or frame.size == 0:
            return 0

        # Match the grayscale JAFFE training images.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)
        tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            output = model(tensor)
            prediction = torch.argmax(output, dim=1).item()

        return int(prediction)

    return predict
