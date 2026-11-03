
from pathlib import Path
import csv
import re
from collections import Counter

# --------------------------------------------------
# 1. FIND THE PROJECT FOLDERS
# --------------------------------------------------

# Project's main folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Folder containing the original JAFFE images
DATASET_DIR = PROJECT_ROOT / "src" / "data" / "external" / "jaffe"

# Folder where our CSV manifest will be saved
OUTPUT_DIR = PROJECT_ROOT / "src" / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "jaffe_manifest.csv"


# --------------------------------------------------
# 2. DEFINE OUR THREE EMOTIONS
# --------------------------------------------------

EMOTION_LABELS = {
    "HA": "Happy",
    "SA": "Sad",
    "SU": "Surprise"
}


# --------------------------------------------------
# 3. CHECK THAT THE DATASET FOLDER EXISTS
# --------------------------------------------------

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"JAFFE folder not found: {DATASET_DIR}"
    )

# Create the processed folder if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 4. READ THE IMAGE FILENAMES
# --------------------------------------------------

# Example filename: KA.HA1.29.tiff
filename_pattern = re.compile(
    r"^([A-Z]{2})\.([A-Z]{2})([1-5])\.(\d+)\.tiff$",
    re.IGNORECASE
)

image_files = sorted(
    file for file in DATASET_DIR.rglob("*")
    if file.is_file() and file.suffix.lower() in [".tiff", ".tif"]
)

print("\n========== JAFFE DATASET ==========")

print(f"Total TIFF images found: {len(image_files)}")


# --------------------------------------------------
# 5. SELECT HAPPY, SAD AND SURPRISE
# --------------------------------------------------

dataset_records = []
emotion_counts = Counter()

excluded_count = 0
unrecognized_files = []

for image_path in image_files:

    filename = image_path.name

    match = filename_pattern.match(filename)

    if not match:
        unrecognized_files.append(filename)
        continue

    participant = match.group(1).upper()
    emotion_code = match.group(2).upper()
    pose_number = match.group(3)
    image_number = match.group(4)

    # Ignore emotions other than our three classes
    if emotion_code not in EMOTION_LABELS:
        excluded_count += 1
        continue

    emotion_name = EMOTION_LABELS[emotion_code]

    dataset_records.append({
        "image_path": image_path.relative_to(
            PROJECT_ROOT
        ).as_posix(),

        "participant": participant,
        "emotion_code": emotion_code,
        "label": emotion_name,
        "pose_number": pose_number,
        "image_number": image_number
    })

    emotion_counts[emotion_name] += 1


# --------------------------------------------------
# 6. DISPLAY THE RESULTS
# --------------------------------------------------

print("\n========== SELECTED EMOTIONS ==========")

for emotion in ["Happy", "Sad", "Surprise"]:
    print(f"{emotion}: {emotion_counts[emotion]} images")

print("---------------------------------------")

print(f"Selected images: {len(dataset_records)}")
print(f"Excluded images: {excluded_count}")
print(f"Unrecognized filenames: {len(unrecognized_files)}")


# --------------------------------------------------
# 7. CHECK THE DATASET
# --------------------------------------------------

if len(dataset_records) == 0:
    raise ValueError("No matching JAFFE images were found.")

if any(emotion_counts[e] == 0 for e in EMOTION_LABELS.values()):
    raise ValueError(
        "One or more emotion categories have no images."
    )

if unrecognized_files:
    print("\nWARNING: Some filenames were not recognised:")
    for filename in unrecognized_files:
        print(filename)


# --------------------------------------------------
# 8. CREATE THE CSV MANIFEST
# --------------------------------------------------

fieldnames = [
    "image_path",
    "participant",
    "emotion_code",
    "label",
    "pose_number",
    "image_number"
]

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(dataset_records)


# --------------------------------------------------
# 9. FINAL MESSAGE
# --------------------------------------------------

print("\n========== SUCCESS ==========")

print("JAFFE dataset preparation completed!")

print(f"CSV saved at: {OUTPUT_FILE}")

print("\nYour three classes are:")
print("1. Happy")
print("2. Sad")
print("3. Surprise")

print("\nYou are ready for the next stage!")
