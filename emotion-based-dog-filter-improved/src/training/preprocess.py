
from pathlib import Path
import csv
from PIL import Image
from collections import Counter

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_FILE = (
    PROJECT_ROOT / "src" / "data" /
    "processed" / "jaffe_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "src" / "data" /
    "processed" / "images"
)

OUTPUT_MANIFEST = (
    PROJECT_ROOT / "src" / "data" /
    "processed" / "jaffe_split_manifest.csv"
)

IMAGE_SIZE = (224, 224)


# --------------------------------------------------
# 2. PARTICIPANT-BASED DATA SPLIT
# --------------------------------------------------

# All images from one participant stay in one group.
# This prevents the same person's images appearing
# in both training and testing.

SPLITS = {
    "train": ["MK", "KA", "TM", "YM", "KM", "NA"],
    "validation": ["KR", "KL"],
    "test": ["NM", "UY"]
}


# --------------------------------------------------
# 3. CHECK THE MANIFEST
# --------------------------------------------------

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(
        f"Manifest not found: {MANIFEST_FILE}"
    )

with open(
    MANIFEST_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)
    records = list(reader)

if not records:
    raise ValueError("The dataset manifest is empty.")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 4. CHECK PARTICIPANT ASSIGNMENTS
# --------------------------------------------------

participant_to_split = {}

for split_name, participants in SPLITS.items():
    for participant in participants:

        if participant in participant_to_split:
            raise ValueError(
                f"Participant {participant} appears in "
                "more than one split."
            )

        participant_to_split[participant] = split_name


# --------------------------------------------------
# 5. PREPROCESS EACH IMAGE
# --------------------------------------------------

processed_records = []
split_counts = Counter()
class_counts = Counter()

print("\n========== IMAGE PREPROCESSING ==========")

for record in records:

    original_path = PROJECT_ROOT / record["image_path"]

    participant = record["participant"]
    label = record["label"]

    if participant not in participant_to_split:
        raise ValueError(
            f"Participant not assigned to a split: {participant}"
        )

    if not original_path.exists():
        raise FileNotFoundError(
            f"Image not found: {original_path}"
        )

    split = participant_to_split[participant]

    # Create a separate folder for each split and emotion
    destination_folder = OUTPUT_DIR / split / label
    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_filename = original_path.stem + ".png"
    output_path = destination_folder / output_filename

    # Open original TIFF image
    with Image.open(original_path) as image:

        # Convert grayscale image to RGB
        image = image.convert("RGB")

        # Resize image to 224 x 224
        image = image.resize(
            IMAGE_SIZE,
            Image.Resampling.LANCZOS
        )

        # Save processed image
        image.save(output_path, format="PNG")

    processed_records.append({
        "processed_image_path": output_path.relative_to(
            PROJECT_ROOT
        ).as_posix(),

        "original_image_path": record["image_path"],
        "participant": participant,
        "label": label,
        "split": split
    })

    split_counts[split] += 1
    class_counts[(split, label)] += 1

    print(f"Processed: {original_path.name} -> {split}/{label}")


# --------------------------------------------------
# 6. SAVE THE NEW CSV
# --------------------------------------------------

fieldnames = [
    "processed_image_path",
    "original_image_path",
    "participant",
    "label",
    "split"
]

with open(
    OUTPUT_MANIFEST,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(processed_records)


# --------------------------------------------------
# 7. DISPLAY RESULTS
# --------------------------------------------------

print("\n========== PREPROCESSING RESULTS ==========")

print(f"Total processed images: {len(processed_records)}")

for split in ["train", "validation", "test"]:

    print(f"\n{split.upper()}: {split_counts[split]} images")

    for label in ["Happy", "Sad", "Surprise"]:
        print(
            f"  {label}: {class_counts[(split, label)]}"
        )

print("\n============================================")

print("Image preprocessing completed successfully!")

print(f"\nProcessed images saved in:\n{OUTPUT_DIR}")

print(f"\nNew CSV saved at:\n{OUTPUT_MANIFEST}")
