import argparse
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input

from sign_language_utils import FEATURE_VECTOR_SIZE


def load_pickle_dataset(path):
    with path.open("rb") as input_file:
        dataset = pickle.load(input_file)

    data = np.asarray(dataset["data"], dtype=np.float32)
    labels = np.asarray(dataset["labels"])

    if data.size == 0 or labels.size == 0:
        raise ValueError(f"Dataset gol în {path}. Rulează mai întâi extracția landmark-urilor.")

    if data.ndim != 2 or data.shape[1] != FEATURE_VECTOR_SIZE:
        raise ValueError(
            f"Format invalid pentru {path}: așteptat (N, {FEATURE_VECTOR_SIZE}), primit {data.shape}"
        )

    if data.shape[0] != labels.shape[0]:
        raise ValueError(
            f"Număr diferit de exemple și etichete în {path}: {data.shape[0]} vs {labels.shape[0]}"
        )

    return data, labels


def build_model(num_classes):
    return Sequential(
        [
            Input(shape=(FEATURE_VECTOR_SIZE,)),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Antrenează modelul pentru recunoașterea alfabetului limbajului semnelor."
    )
    parser.add_argument("--train-data", type=Path, default=Path("data/processed/train_data.pickle"))
    parser.add_argument("--test-data", type=Path, default=Path("data/processed/test_data.pickle"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-output", type=Path, default=Path("models/sign_language_model.h5"))
    parser.add_argument("--labels-output", type=Path, default=Path("models/label_binarizer.pickle"))
    return parser.parse_args()


def main():
    args = parse_args()

    train_x, train_labels = load_pickle_dataset(args.train_data)
    test_x, test_labels = load_pickle_dataset(args.test_data)

    label_binarizer = LabelBinarizer()
    train_y = label_binarizer.fit_transform(train_labels)
    test_y = label_binarizer.transform(test_labels)

    class_names = label_binarizer.classes_
    if len(class_names) != 24:
        print(
            f"[WARN] Numărul de clase detectate este {len(class_names)}, "
            "nu 24. Modelul va fi construit pe baza claselor disponibile."
        )

    model = build_model(len(class_names))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        patience=10,
        mode="max",
        restore_best_weights=True,
        verbose=1,
    )

    model.summary()
    model.fit(
        train_x,
        train_y,
        validation_data=(test_x, test_y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    evaluation = model.evaluate(test_x, test_y, verbose=0)
    print(f"[INFO] Test loss: {evaluation[0]:.4f}")
    print(f"[INFO] Test accuracy: {evaluation[1]:.4f}")

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)

    model.save(args.model_output)
    with args.labels_output.open("wb") as output_file:
        pickle.dump(label_binarizer, output_file)

    print(f"[INFO] Model salvat în: {args.model_output}")
    print(f"[INFO] LabelBinarizer salvat în: {args.labels_output}")


if __name__ == "__main__":
    main()
