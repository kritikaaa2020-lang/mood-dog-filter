
import csv
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Three emotion classes used in our project
EMOTION_CLASSES = ["Happy", "Sad", "Surprise"]

# Convert emotion names into numerical labels
LABEL_MAP = {
    "Happy": 0,
    "Sad": 1,
    "Surprise": 2
}


# Transformations applied to training images
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Transformations applied to validation and testing images
evaluation_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Custom JAFFE dataset class
class JAFFEDataset(Dataset):

    def __init__(self, csv_file, split, transform=None):
        csv_path = Path(csv_file).resolve()
        if not csv_path.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {csv_path}")

        with csv_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required_columns = {"processed_image_path", "label", "split"}
            if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
                raise ValueError(
                    f"Dataset manifest must contain columns: "
                    f"{', '.join(sorted(required_columns))}"
                )
            self.data = [
                row for row in reader
                if row["split"] == split
            ]

        if not self.data:
            raise ValueError(f"No images found for the '{split}' split.")

        for row in self.data:
            if row["label"] not in LABEL_MAP:
                raise ValueError(f"Unknown emotion label: {row['label']}")

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Get the selected row
        row = self.data[index]

        full_image_path = PROJECT_ROOT / row["processed_image_path"]
        if not full_image_path.is_file():
            raise FileNotFoundError(f"Dataset image not found: {full_image_path}")

        # Open image and convert it to RGB
        image = Image.open(full_image_path).convert("RGB")

        # Convert emotion name into numerical label
        label = LABEL_MAP[row["label"]]

        # Apply image transformations
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# Function to create training, validation and testing loaders
def create_data_loaders(csv_file, batch_size=8):

    train_dataset = JAFFEDataset(
        csv_file,
        "train",
        train_transform
    )

    validation_dataset = JAFFEDataset(
        csv_file,
        "validation",
        evaluation_transform
    )

    test_dataset = JAFFEDataset(
        csv_file,
        "test",
        evaluation_transform
    )

    # Create training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(42)
    )

    # Create validation loader
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Create testing loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, validation_loader, test_loader


# Test the dataset loader
if __name__ == "__main__":

    csv_file = (
        PROJECT_ROOT / "src" / "data" / "processed"
        / "jaffe_split_manifest.csv"
    )

    train_loader, validation_loader, test_loader = (
        create_data_loaders(csv_file)
    )

    print("Dataset loaders created successfully!")

    print("\nDataset batches:")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(validation_loader))
    print("Testing batches:", len(test_loader))

    # Load one batch of training images
    images, labels = next(iter(train_loader))

    print("\nBatch information:")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print("\nNumerical labels:", labels.tolist())
    print("Emotion classes:", EMOTION_CLASSES)
