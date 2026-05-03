import argparse
import pickle
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from sign_language_utils import FEATURE_VECTOR_SIZE, compute_relative_landmarks


BUFFER_SIZE = 15
STABILITY_SECONDS = 0.5
INFERENCE_WIDTH = 640


def draw_label(frame, label):
    cv2.putText(
        frame,
        f"Litera: {label}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )


def draw_status(frame, status_message, color):
    cv2.putText(
        frame,
        status_message,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def resize_for_inference(frame):
    height, width = frame.shape[:2]
    if width <= INFERENCE_WIDTH:
        return frame

    scale = INFERENCE_WIDTH / float(width)
    resized_height = max(1, int(height * scale))
    return cv2.resize(frame, (INFERENCE_WIDTH, resized_height), interpolation=cv2.INTER_AREA)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recunoaștere în timp real a limbajului semnelor folosind webcam-ul."
    )
    parser.add_argument("--model", type=Path, default=Path("models/sign_language_model.h5"))
    parser.add_argument("--labels", type=Path, default=Path("models/label_binarizer.pickle"))
    parser.add_argument("--camera-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Modelul nu există: {args.model}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Fișierul de etichete nu există: {args.labels}")

    model = tf.keras.models.load_model(args.model)
    with args.labels.open("rb") as input_file:
        label_binarizer = pickle.load(input_file)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Webcam-ul nu a putut fi deschis.")

    prediction_buffer = deque(maxlen=BUFFER_SIZE)
    displayed_label = "-"
    candidate_label = None
    candidate_start_time = None
    status_message = "Nicio mana detectata"
    status_color = (0, 0, 255)

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands_detector:
            while True:
                success, frame = cap.read()
                if not success:
                    print("[WARN] Cadru indisponibil de la webcam.")
                    break

                frame = cv2.flip(frame, 1)
                inference_frame = resize_for_inference(frame)
                current_prediction = None
                hand_landmarks = None

                try:
                    rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                    results = hands_detector.process(rgb_frame)
                except Exception as exc:
                    status_message = f"Eroare MediaPipe: {exc}"
                    status_color = (0, 165, 255)
                    results = None

                if results and results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    features = compute_relative_landmarks(hand_landmarks)

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    if features is not None and len(features) == FEATURE_VECTOR_SIZE:
                        try:
                            feature_array = np.asarray(features, dtype=np.float32).reshape(1, -1)
                            prediction = model.predict(feature_array, verbose=0)[0]
                            predicted_index = int(np.argmax(prediction))
                            current_prediction = label_binarizer.classes_[predicted_index]
                            prediction_buffer.append(current_prediction)
                            status_message = "Mana detectata"
                            status_color = (0, 255, 0)
                        except Exception as exc:
                            status_message = f"Eroare predictie: {exc}"
                            status_color = (0, 165, 255)
                    else:
                        status_message = "Landmark-uri invalide"
                        status_color = (0, 165, 255)
                else:
                    prediction_buffer.clear()
                    candidate_label = None
                    candidate_start_time = None
                    status_message = "Nicio mana detectata"
                    status_color = (0, 0, 255)

                if prediction_buffer:
                    most_common_label, _ = Counter(prediction_buffer).most_common(1)[0]
                    now = time.monotonic()

                    if most_common_label != displayed_label:
                        if candidate_label != most_common_label:
                            candidate_label = most_common_label
                            candidate_start_time = now
                        elif candidate_start_time and (now - candidate_start_time) >= STABILITY_SECONDS:
                            displayed_label = most_common_label
                            candidate_label = None
                            candidate_start_time = None
                    else:
                        candidate_label = None
                        candidate_start_time = None

                draw_label(frame, displayed_label)
                draw_status(frame, status_message, status_color)

                cv2.imshow("Sign Language Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
