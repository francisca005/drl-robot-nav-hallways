import numpy as np


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


def extract_lidar_features(lidar: np.ndarray, previous_action: int) -> np.ndarray:
    """
    Converts raw 360-degree LiDAR readings into a compact feature vector.

    Feature vector:
    0. min_front
    1. mean_front
    2. left_clearance
    3. right_clearance
    4. clearance_asymmetry
    5. max_gap_center
    6. max_gap_width
    7. rear_clearance
    8. min_left
    9. min_right
    10. min_rear
    11. previous_action normalized
    """
    lidar_norm = normalize_lidar(lidar)

    # Same approximate sector logic used by the current reward function.
    left = lidar_norm[100:170]
    front = lidar_norm[170:190]
    right = lidar_norm[190:260]

    # Rear sector: combine end and beginning of circular scan.
    rear = np.concatenate([lidar_norm[0:40], lidar_norm[320:360]])

    min_front = np.min(front)
    mean_front = np.mean(front)

    left_clearance = np.mean(left)
    right_clearance = np.mean(right)

    clearance_asymmetry = right_clearance - left_clearance

    max_gap_center, max_gap_width = longest_free_gap(lidar_norm)

    rear_clearance = np.mean(rear)

    min_left = np.min(left)
    min_right = np.min(right)
    min_rear = np.min(rear)

    previous_action_norm = float(previous_action) / 5.0

    features = np.array(
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

    return features