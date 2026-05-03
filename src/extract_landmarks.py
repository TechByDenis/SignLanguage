import argparse
import pickle
from pathlib import Path

import cv2
import mediapipe as mp

from sign_language_utils import FEATURE_VECTOR_SIZE, extract_hand_features


DATASET_SPLITS = ("Train", "Test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def extract_sample(image_path, hands_detector):
    try:
        image = cv2.imread(str(image_path))
    except Exception as exc:
        print(f"[WARN] Eroare la citirea imaginii {image_path}: {exc}")
        return None

    if image is None:
        print(f"[WARN] Imagine invalidă sau inaccesibilă: {image_path}")
        return None

    sample = extract_hand_features(image, hands_detector)
    if sample is None or len(sample) != FEATURE_VECTOR_SIZE:
        return None

    return sample


def process_split(split_dir, output_path):
    data = []
    labels = []
    skipped = 0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands_detector:
        class_dirs = sorted(
            folder for folder in split_dir.iterdir() if folder.is_dir()
        )

        for class_dir in class_dirs:
            label = class_dir.name
            image_paths = sorted(
                path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
            )
            print(f"[INFO] Procesez {split_dir.name}/{label}: {len(image_paths)} imagini")

            for image_path in image_paths:
                sample = extract_sample(image_path, hands_detector)
                if sample is None:
                    skipped += 1
                    continue
                data.append(sample)
                labels.append(label)

    with output_path.open("wb") as output_file:
        pickle.dump({"data": data, "labels": labels}, output_file)

    print(
        f"[INFO] Salvate {len(data)} exemple în {output_path.name}. "
        f"Exemple sărite: {skipped}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extrage landmark-uri MediaPipe pentru datasetul de limbajul semnelor."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/semne"),
        help="Directorul rădăcină care conține Train și Test.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path("data/processed/train_data.pickle"),
        help="Fișierul pickle rezultat pentru datele de train.",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/processed/test_data.pickle"),
        help="Fișierul pickle rezultat pentru datele de test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Datasetul nu există: {dataset_root}")

    for split_name in DATASET_SPLITS:
        split_dir = dataset_root / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Lipsește directorul pentru split: {split_dir}")

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.test_output.parent.mkdir(parents=True, exist_ok=True)

    process_split(dataset_root / "Train", args.train_output)
    process_split(dataset_root / "Test", args.test_output)


if __name__ == "__main__":
    main()
