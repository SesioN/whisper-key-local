import numpy as np


def segment_distance(x, y, ax, ay, bx, by):
    px, py = x - ax, y - ay
    vx, vy = bx - ax, by - ay
    h = np.clip((px * vx + py * vy) / (vx * vx + vy * vy), 0.0, 1.0)
    qx, qy = px - vx * h, py - vy * h
    return np.sqrt(qx * qx + qy * qy)


def ring_distance(x, y, cx, cy, radius):
    return np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - radius)


def arc_distance(x, y, cx, cy, radius, lower_only=True):
    ring = ring_distance(x, y, cx, cy, radius)
    if not lower_only:
        return ring
    left = np.sqrt((x - (cx - radius)) ** 2 + (y - cy) ** 2)
    right = np.sqrt((x - (cx + radius)) ** 2 + (y - cy) ** 2)
    return np.where(y >= cy, ring, np.minimum(left, right))


def stroke(distance, half_width, pixel):
    return np.clip((half_width - distance) / pixel + 0.5, 0.0, 1.0).astype(np.float32)


def ring_badge(x, y, pixel):
    return stroke(ring_distance(x, y, 0.0, 0.0, 0.40), 0.018, pixel)


def mic_slash(x, y, pixel):
    body = segment_distance(x, y, 0.0, -0.17, 0.0, -0.02) - 0.055
    cup = arc_distance(x, y, 0.0, -0.04, 0.115)
    stand = segment_distance(x, y, 0.0, 0.075, 0.0, 0.15)
    base = segment_distance(x, y, -0.07, 0.15, 0.07, 0.15)
    outline = np.minimum(np.minimum(np.abs(body), cup), np.minimum(stand, base))
    slash = segment_distance(x, y, -0.17, -0.20, 0.17, 0.20)
    gap = stroke(slash, 0.045, pixel)
    return np.maximum(stroke(outline, 0.016, pixel) * (1.0 - gap), stroke(slash, 0.016, pixel))


def check_mark(x, y, pixel):
    d = np.minimum(segment_distance(x, y, -0.15, 0.0, -0.045, 0.11),
                   segment_distance(x, y, -0.045, 0.11, 0.16, -0.10))
    return stroke(d, 0.035, pixel)


def exclamation(x, y, pixel):
    bar = segment_distance(x, y, 0.0, -0.17, 0.0, 0.05)
    dot = np.sqrt(x * x + (y - 0.15) ** 2)
    return np.maximum(stroke(bar, 0.038, pixel), stroke(dot, 0.042, pixel))
