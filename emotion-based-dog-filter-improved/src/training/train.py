
import os
import sys
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Get the project folder
PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Add src folder to Python path
sys.path.append(os.path.join(PROJECT_DIR, "src"))

from models.emotion_model import create_emotion_model
from training.dataset_loader import EMOTION_CLASSES, create_data_loaders


# -----------------------------
# 1. PROJECT SETTINGS
# -----------------------------

DEVICE = torch.device("cpu")

EPOCHS = 40
BATCH_SIZE = 8
PATIENCE = 8
SEED = 42

CSV_FILE = os.path.join(
    PROJECT_DIR,
    "src",
    "data",
    "processed",
    "jaffe_split_manifest.csv"
)

MODEL_FOLDER = os.path.join(
    PROJECT_DIR,
    "src",
    "models"
)

MODEL_SAVE_PATH = os.path.join(
    MODEL_FOLDER,
    "jaffe_emotion_model.pth"
)


# -----------------------------
# 2. LOAD DATASET
# -----------------------------

train_loader, validation_loader, _ = (
    create_data_loaders(
        CSV_FILE,
        batch_size=BATCH_SIZE
    )
)

print("Dataset loaded successfully!")


# -----------------------------
# 3. CREATE MODEL
# -----------------------------

parser = argparse.ArgumentParser(description="Fine-tune the JAFFE emotion model.")
parser.add_argument(
    "--output",
    default=MODEL_SAVE_PATH,
    help="Where to save the best validation checkpoint."
)
ARGS = parser.parse_args()

random.seed(SEED)
torch.manual_seed(SEED)

model = create_emotion_model(pretrained=not os.path.exists(MODEL_SAVE_PATH))
if os.path.exists(MODEL_SAVE_PATH):
    checkpoint = torch.load(
        MODEL_SAVE_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(checkpoint)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.AdamW(
    [
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 3e-4}
    ],
    weight_decay=1e-3
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)



# -----------------------------
# 4. TRAINING FUNCTION
# -----------------------------

def train_one_epoch(model, loader):

    model.train()
    # Keep pretrained BatchNorm running statistics stable on this small dataset.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()

    total_loss = 0
    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -----------------------------
# 5. VALIDATION FUNCTION
# -----------------------------

def validate_model(model, loader):

    model.eval()

    total_loss = 0
    confusion = torch.zeros(
        len(EMOTION_CLASSES),
        len(EMOTION_CLASSES),
        dtype=torch.int64
    )

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            for actual, predicted in zip(labels, predictions):
                confusion[actual, predicted] += 1

    recalls = confusion.diag().float() / confusion.sum(dim=1).clamp_min(1)
    balanced_accuracy = recalls.mean().item() * 100
    average_loss = total_loss / len(loader)
    return average_loss, balanced_accuracy


# -----------------------------
# 6. TRAINING LOOP
# -----------------------------

best_validation_accuracy = -1
best_validation_loss = float("inf")
epochs_without_improvement = 0

print("\nStarting model training...\n")

for epoch in range(EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
    )

    validation_loss, validation_accuracy = validate_model(
        model,
        validation_loader
    )

    print(f"Epoch [{epoch + 1}/{EPOCHS}]")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {validation_loss:.4f}")
    print(f"Validation Balanced Accuracy: {validation_accuracy:.2f}%")
    print("-" * 40)

    scheduler.step(validation_loss)

    improved = (
        validation_accuracy > best_validation_accuracy
        or (
            validation_accuracy == best_validation_accuracy
            and validation_loss < best_validation_loss
        )
    )
    if improved:
        best_validation_accuracy = validation_accuracy
        best_validation_loss = validation_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            ARGS.output
        )

        print("Best model saved!")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping after {epoch + 1} epochs.")
        break


print("\nTraining completed!")
print("Best validation balanced accuracy:",
      f"{best_validation_accuracy:.2f}%")

print("Model saved at:")
print(ARGS.output)
