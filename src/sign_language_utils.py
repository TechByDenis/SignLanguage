from typing import Optional

import cv2


FEATURE_VECTOR_SIZE = 42


def compute_relative_landmarks(hand_landmarks) -> Optional[list[float]]:
    """Returnează 42 valori normalizate identic pentru train și inference."""
    if hand_landmarks is None or not getattr(hand_landmarks, "landmark", None):
        return None

    points = [(landmark.x, landmark.y) for landmark in hand_landmarks.landmark]
    if len(points) != 21:
        return None

    wrist_x, wrist_y = points[0]
    relative_points = []
    max_distance = 0.0

    for x_coord, y_coord in points:
        rel_x = x_coord - wrist_x
        rel_y = y_coord - wrist_y
        relative_points.append((rel_x, rel_y))
        max_distance = max(max_distance, abs(rel_x), abs(rel_y))

    if max_distance <= 0.0:
        return None

    normalized = []
    for rel_x, rel_y in relative_points:
        normalized.extend([rel_x / max_distance, rel_y / max_distance])

    return normalized if len(normalized) == FEATURE_VECTOR_SIZE else None


def extract_hand_features(bgr_image, hands_detector) -> Optional[list[float]]:
    """Extrage landmark-uri dintr-o imagine BGR și aplică normalizarea comună."""
    if bgr_image is None:
        return None

    try:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_image)
    except Exception as exc:
        print(f"[WARN] MediaPipe a eșuat la procesarea cadrului: {exc}")
        return None

    if not results or not results.multi_hand_landmarks:
        return None

    return compute_relative_landmarks(results.multi_hand_landmarks[0])
