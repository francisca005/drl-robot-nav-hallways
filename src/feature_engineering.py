import numpy as np

# Feature dimensions for each variant.
FEATURE_SET_DIMS = {
    "full": 12,
    "reduced": 8,
    "directional": 5,
}


def normalize_lidar(lidar: np.ndarray) -> np.ndarray:
    """
    Clips LiDAR readings to [0, 10] and normalizes them to [0, 1].
    """
    lidar = np.asarray(lidar, dtype=np.float32)
    lidar = np.clip(lidar, 0.0, 10.0)
    return lidar / 10.0


def longest_free_gap(lidar_norm: np.ndarray, threshold: float = 0.55):
    """
    Finds the longest contiguous sector where LiDAR distance is above threshold.

    Returns:
        gap_center_norm: center index normalized to [0, 1]
        gap_width_norm: width normalized to [0, 1]
    """
    free = lidar_norm > threshold

    best_start = 0
    best_len = 0
    current_start = 0
    current_len = 0

    for i, is_free in enumerate(free):
        if is_free:
            if current_len == 0:
                current_start = i
            current_len += 1
        else:
            if current_len > best_len:
                best_start = current_start
                best_len = current_len
            current_len = 0

    if current_len > best_len:
        best_start = current_start
        best_len = current_len

    if best_len == 0:
        return 0.5, 0.0

    center = best_start + best_len / 2.0

    gap_center_norm = center / 359.0
    gap_width_norm = best_len / 360.0

    return gap_center_norm, gap_width_norm


def extract_lidar_features(
    lidar: np.ndarray,
    previous_action: int,
    feature_set: str = "full",
) -> np.ndarray:
    """
    Converts raw 360-degree LiDAR readings into a compact feature vector.

    feature_set options:
        "full"        — 12 features (all, including redundant ones)
        "reduced"     — 8 features (removes mean_front, clearance_asymmetry,
                        rear_clearance, min_rear)
        "directional" — 5 features (only decision-critical directional signals)

    Full feature list (indices in "full"):
        0  min_front            — closest obstacle ahead (safety signal)
        1  mean_front           — average front distance [full only]
        2  left_clearance       — mean lateral space on left
        3  right_clearance      — mean lateral space on right
        4  clearance_asymmetry  — right - left [full and directional]
        5  max_gap_center       — direction of largest free corridor
        6  max_gap_width        — width of largest free corridor
        7  rear_clearance       — mean rear distance [full only]
        8  min_left             — closest obstacle on left
        9  min_right            — closest obstacle on right
        10 min_rear             — closest obstacle behind [full only]
        11 previous_action      — last action taken (normalized)

    "reduced" keeps: min_front, left_clearance, right_clearance,
                     max_gap_center, max_gap_width, min_left, min_right,
                     previous_action
    "directional" keeps: min_front, clearance_asymmetry, max_gap_center,
                         max_gap_width, previous_action
    """
    lidar_norm = normalize_lidar(lidar)

    left = lidar_norm[100:170]
    front = lidar_norm[170:190]
    right = lidar_norm[190:260]
    rear = np.concatenate([lidar_norm[0:40], lidar_norm[320:360]])

    min_front = float(np.min(front))
    left_clearance = float(np.mean(left))
    right_clearance = float(np.mean(right))
    clearance_asymmetry = right_clearance - left_clearance
    max_gap_center, max_gap_width = longest_free_gap(lidar_norm)
    previous_action_norm = float(previous_action) / 5.0

    if feature_set == "directional":
        return np.array(
            [
                min_front,
                clearance_asymmetry,
                max_gap_center,
                max_gap_width,
                previous_action_norm,
            ],
            dtype=np.float32,
        )

    min_left = float(np.min(left))
    min_right = float(np.min(right))

    if feature_set == "reduced":
        return np.array(
            [
                min_front,
                left_clearance,
                right_clearance,
                max_gap_center,
                max_gap_width,
                min_left,
                min_right,
                previous_action_norm,
            ],
            dtype=np.float32,
        )

    # "full": all 12 features
    mean_front = float(np.mean(front))
    rear_clearance = float(np.mean(rear))
    min_rear = float(np.min(rear))

    return np.array(
        [
            min_front,
            mean_front,
            left_clearance,
            right_clearance,
            clearance_asymmetry,
            max_gap_center,
            max_gap_width,
            rear_clearance,
            min_left,
            min_right,
            min_rear,
            previous_action_norm,
        ],
        dtype=np.float32,
    )
