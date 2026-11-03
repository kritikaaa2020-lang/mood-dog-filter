import os
import sys
import argparse
import torch

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.append(os.path.join(PROJECT_DIR, "src"))

from models.emotion_model import create_emotion_model
from training.dataset_loader import create_data_loaders

DEVICE = torch.device("cpu")

CSV_PATH = os.path.join(
    PROJECT_DIR,
    "src/data/processed/jaffe_split_manifest.csv"
)

DEFAULT_MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "src/models/jaffe_emotion_model.pth"
)

CLASSES = ["Happy", "Sad", "Surprise"]

parser = argparse.ArgumentParser(description="Evaluate the JAFFE emotion model.")
parser.add_argument(
    "--model",
    default=DEFAULT_MODEL_PATH,
    help="Checkpoint to evaluate."
)
args = parser.parse_args()

_, _, test_loader = create_data_loaders(CSV_PATH)

model = create_emotion_model()
model.load_state_dict(
    torch.load(args.model, map_location=DEVICE, weights_only=True)
)
model.to(DEVICE)
model.eval()

confusion = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)

        predictions = outputs.argmax(dim=1)
        for actual, predicted in zip(labels, predictions):
            confusion[actual, predicted] += 1

print("\nConfusion matrix (rows = actual, columns = predicted):")
print(confusion.tolist())

print("\nPer-class results:")
for index, emotion in enumerate(CLASSES):
    actual_count = confusion[index].sum().item()
    predicted_count = confusion[:, index].sum().item()
    recall = confusion[index, index].item() / max(actual_count, 1)
    precision = confusion[index, index].item() / max(predicted_count, 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )
    print(
        f"{emotion}: precision={precision:.2%}, "
        f"recall={recall:.2%}, F1={f1:.2%} "
        f"({int(confusion[index, index])}/{actual_count})"
    )

total = confusion.sum().item()
accuracy = confusion.diag().sum().item() / max(total, 1)
print(f"\nTest accuracy: {accuracy:.2%} ({int(confusion.diag().sum())}/{total})")