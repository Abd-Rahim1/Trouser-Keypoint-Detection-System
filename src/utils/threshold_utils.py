def calculate_visibility(keypoints, threshold):
    """Calculate visibility based on threshold"""
    visible_count = 0
    for x, y, v in keypoints:
        if v > threshold:
            visible_count += 1
    return visible_count

def filter_keypoints_by_confidence(raw_keypoints, threshold):
    """Filter keypoints by confidence threshold"""
    filtered = []
    for kp in raw_keypoints:
        x, y, conf = kp
        visible = 1 if conf > threshold else 0
        filtered.append((x, y, visible))
    return filtered