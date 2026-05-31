from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="BrailleVision API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   "https://braille-vision2026-web.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ROI_SCALE = 0.70
MIN_CONTOUR_AREA = 15.0
MAX_CONTOUR_AREA = 350.0
MIN_CIRCULARITY = 0.60
MIN_ASPECT_RATIO = 0.70
MAX_ASPECT_RATIO = 1.30


GRADE_1_MAP: Dict[Tuple[int, int, int, int, int, int], str] = {
    (1, 0, 0, 0, 0, 0): "a",
    (1, 1, 0, 0, 0, 0): "b",
    (1, 0, 0, 1, 0, 0): "c",
    (1, 0, 0, 1, 1, 0): "d",
    (1, 0, 0, 0, 1, 0): "e",
    (1, 1, 0, 1, 0, 0): "f",
    (1, 1, 0, 1, 1, 0): "g",
    (1, 1, 0, 0, 1, 0): "h",
    (0, 1, 0, 1, 0, 0): "i",
    (0, 1, 0, 1, 1, 0): "j",
    (1, 0, 1, 0, 0, 0): "k",
    (1, 1, 1, 0, 0, 0): "l",
    (1, 0, 1, 1, 0, 0): "m",
    (1, 0, 1, 1, 1, 0): "n",
    (1, 0, 1, 0, 1, 0): "o",
    (1, 1, 1, 1, 0, 0): "p",
    (1, 1, 1, 1, 1, 0): "q",
    (1, 1, 1, 0, 1, 0): "r",
    (0, 1, 1, 1, 0, 0): "s",
    (0, 1, 1, 1, 1, 0): "t",
    (1, 0, 1, 0, 0, 1): "u",
    (1, 1, 1, 0, 0, 1): "v",
    (0, 1, 0, 1, 1, 1): "w",
    (1, 0, 1, 1, 0, 1): "x",
    (1, 0, 1, 1, 1, 1): "y",
    (1, 0, 1, 0, 1, 1): "z",
    (0, 0, 0, 0, 0, 0): " ",
}


@dataclass
class DotCandidate:
    x: int
    y: int
    w: int
    h: int
    area: float
    circularity: float
    contrast: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def diameter(self) -> float:
        return float((self.w + self.h) / 2.0)


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    matrix = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(matrix, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Unable to decode image frame.")
    return frame


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def _center_roi_bounds(width: int, height: int) -> Tuple[int, int, int, int]:
    roi_width = int(width * ROI_SCALE)
    roi_height = int(height * ROI_SCALE)
    left = max(0, (width - roi_width) // 2)
    top = max(0, (height - roi_height) // 2)
    right = min(width, left + roi_width)
    bottom = min(height, top + roi_height)
    return left, top, right, bottom


def _encode_debug_image(mask: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        return ""
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _extract_dot_candidates(binary: np.ndarray, grayscale: np.ndarray) -> List[DotCandidate]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[DotCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0.0:
            continue

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < MIN_CIRCULARITY:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0:
            continue

        aspect_ratio = float(w) / float(h)
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue

        contour_mask = np.zeros(grayscale.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        contour_pixels = grayscale[contour_mask == 255]
        if contour_pixels.size == 0:
            continue

        outer_mask = np.zeros(grayscale.shape, dtype=np.uint8)
        radius = max(4, int(max(w, h) * 1.8))
        cv2.circle(outer_mask, (int(x + w / 2), int(y + h / 2)), radius, 255, thickness=-1)
        ring_mask = cv2.subtract(outer_mask, contour_mask)
        ring_pixels = grayscale[ring_mask == 255]
        background_mean = float(np.mean(ring_pixels)) if ring_pixels.size else float(np.mean(grayscale))
        foreground_mean = float(np.mean(contour_pixels))
        contrast = max(0.0, background_mean - foreground_mean)

        candidates.append(
            DotCandidate(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=area,
                circularity=circularity,
                contrast=contrast,
            )
        )

    candidates.sort(key=lambda dot: (dot.y, dot.x))
    return candidates


def _cluster_dots(candidates: List[DotCandidate], diameter: float) -> List[List[DotCandidate]]:
    if not candidates:
        return []

    horizontal_radius = max(3.5 * diameter, 18.0)
    vertical_radius = max(4.0 * diameter, 28.0)
    visited = [False] * len(candidates)
    clusters: List[List[DotCandidate]] = []

    for start in range(len(candidates)):
        if visited[start]:
            continue

        queue = [start]
        visited[start] = True
        cluster: List[DotCandidate] = []

        while queue:
            current_index = queue.pop()
            current = candidates[current_index]
            cluster.append(current)
            cx, cy = current.center

            for neighbor_index, neighbor in enumerate(candidates):
                if visited[neighbor_index]:
                    continue

                nx, ny = neighbor.center
                if abs(nx - cx) <= horizontal_radius and abs(ny - cy) <= vertical_radius:
                    visited[neighbor_index] = True
                    queue.append(neighbor_index)

        clusters.append(sorted(cluster, key=lambda dot: (dot.y, dot.x)))

    return clusters


def _estimate_orientation(cluster: List[DotCandidate], diameter: float) -> Tuple[np.ndarray, np.ndarray]:
    points = np.array([dot.center for dot in cluster], dtype=np.float32)
    if len(points) < 2:
        return np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)

    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    major = major / (np.linalg.norm(major) + 1e-6)
    if major[0] < 0:
        major = -major

    minor = np.array([-major[1], major[0]], dtype=np.float32)
    if minor[1] < 0:
        minor = -minor

    horizontal_vectors: List[np.ndarray] = []
    vertical_vectors: List[np.ndarray] = []
    target = 2.5 * diameter

    for left_index, left_dot in enumerate(cluster):
        lx, ly = left_dot.center
        for right_dot in cluster[left_index + 1 :]:
            rx, ry = right_dot.center
            vector = np.array([rx - lx, ry - ly], dtype=np.float32)
            length = float(np.linalg.norm(vector))
            if length < diameter or length > target * 1.8:
                continue
            unit = vector / (length + 1e-6)
            if abs(float(np.dot(unit, major))) >= 0.7:
                horizontal_vectors.append(unit if unit[0] >= 0 else -unit)
            if abs(float(np.dot(unit, minor))) >= 0.7:
                candidate = unit if unit[1] >= 0 else -unit
                vertical_vectors.append(candidate)

    if horizontal_vectors:
        averaged = np.mean(horizontal_vectors, axis=0)
        major = averaged / (np.linalg.norm(averaged) + 1e-6)
    if vertical_vectors:
        averaged = np.mean(vertical_vectors, axis=0)
        minor = averaged / (np.linalg.norm(averaged) + 1e-6)
    else:
        minor = np.array([-major[1], major[0]], dtype=np.float32)
        if minor[1] < 0:
            minor = -minor

    return major.astype(np.float32), minor.astype(np.float32)


def _line_groups(clusters: List[List[DotCandidate]], diameter: float) -> List[List[List[DotCandidate]]]:
    if not clusters:
        return []

    sorted_clusters = sorted(
        clusters,
        key=lambda cluster: (
            min(dot.center[1] for dot in cluster),
            min(dot.center[0] for dot in cluster),
        ),
    )
    threshold = max(4.2 * diameter, 34.0)
    lines: List[List[List[DotCandidate]]] = []

    for cluster in sorted_clusters:
        cluster_top = float(min(dot.center[1] for dot in cluster))
        if not lines:
            lines.append([cluster])
            continue

        reference = float(
            np.median([min(dot.center[1] for dot in existing) for existing in lines[-1]])
        )
        if abs(cluster_top - reference) <= threshold:
            lines[-1].append(cluster)
        else:
            lines.append([cluster])

    for line in lines:
        line.sort(key=lambda cluster: min(dot.center[0] for dot in cluster))
    return lines


def _match_pattern(pattern: Sequence[int]) -> Tuple[Tuple[int, int, int, int, int, int], str, int]:
    pattern_tuple = tuple(int(bit) for bit in pattern)
    if pattern_tuple in GRADE_1_MAP:
        return pattern_tuple, GRADE_1_MAP[pattern_tuple], 0

    best_pattern = min(
        GRADE_1_MAP.keys(),
        key=lambda candidate: sum(int(left != right) for left, right in zip(pattern_tuple, candidate)),
    )
    distance = sum(int(left != right) for left, right in zip(pattern_tuple, best_pattern))
    return best_pattern, GRADE_1_MAP[best_pattern], distance


def _decode_cluster(
    cluster: List[DotCandidate],
    diameter: float,
) -> Tuple[str, float, List[int]]:
    anchor_dot = min(cluster, key=lambda dot: (dot.y, dot.x))
    anchor = np.array(anchor_dot.center, dtype=np.float32)
    horizontal_axis, vertical_axis = _estimate_orientation(cluster, diameter)
    step = 2.5 * diameter
    match_radius = 1.45 * diameter

    projected_points = [
        anchor,
        anchor + vertical_axis * step,
        anchor + vertical_axis * (2.0 * step),
        anchor + horizontal_axis * step,
        anchor + horizontal_axis * step + vertical_axis * step,
        anchor + horizontal_axis * step + vertical_axis * (2.0 * step),
    ]

    pattern = [0, 0, 0, 0, 0, 0]
    matched_indices: set[int] = set()
    for node_index, target in enumerate(projected_points):
        best_distance = float("inf")
        best_candidate = -1
        for candidate_index, dot in enumerate(cluster):
            if candidate_index in matched_indices:
                continue
            center = np.array(dot.center, dtype=np.float32)
            distance = float(np.linalg.norm(center - target))
            if distance <= match_radius and distance < best_distance:
                best_distance = distance
                best_candidate = candidate_index

        if best_candidate >= 0:
            matched_indices.add(best_candidate)
            pattern[node_index] = 1

    matched_pattern, translated, hamming_distance = _match_pattern(pattern)
    occupancy = sum(pattern) / 6.0
    geometry = float(np.mean([dot.circularity for dot in cluster]))
    contrast = min(1.0, float(np.mean([dot.contrast for dot in cluster])) / 28.0)
    match_score = max(0.35, 1.0 - (hamming_distance / 6.0))
    confidence = min(
        1.0,
        0.12 + occupancy * 0.28 + geometry * 0.22 + contrast * 0.2 + match_score * 0.18,
    )

    xs = [dot.x for dot in cluster] + [int(point[0] - match_radius) for point in projected_points]
    ys = [dot.y for dot in cluster] + [int(point[1] - match_radius) for point in projected_points]
    xe = [dot.x + dot.w for dot in cluster] + [int(point[0] + match_radius) for point in projected_points]
    ye = [dot.y + dot.h for dot in cluster] + [int(point[1] + match_radius) for point in projected_points]
    box = [
        max(0, int(min(xs))),
        max(0, int(min(ys))),
        max(1, int(max(xe) - min(xs))),
        max(1, int(max(ye) - min(ys))),
    ]

    return translated, confidence, box


def _decode_braille(candidates: List[DotCandidate]) -> Tuple[str, float, List[List[int]]]:
    if not candidates:
        return "", 0.0, []

    diameter = float(np.mean([dot.diameter for dot in candidates]))
    clusters = _cluster_dots(candidates, diameter)
    lines = _line_groups(clusters, diameter)

    text_lines: List[str] = []
    boxes: List[List[int]] = []
    confidences: List[float] = []

    for line in lines:
        row_text = ""
        previous_right: float | None = None

        for cluster in line:
            cluster_left = float(min(dot.x for dot in cluster))
            cluster_right = float(max(dot.x + dot.w for dot in cluster))
            if previous_right is not None and cluster_left - previous_right > diameter * 3.8:
                row_text += " "

            translated, confidence, box = _decode_cluster(cluster, diameter)
            row_text += translated
            boxes.append(box)
            confidences.append(confidence)
            previous_right = cluster_right

        cleaned = row_text.strip()
        if cleaned:
            text_lines.append(cleaned)

    text = "\n".join(text_lines).strip()
    confidence = round(float(np.mean(confidences)), 3) if confidences else 0.0
    return text, confidence, boxes


def process_braille_frame(frame: np.ndarray) -> Tuple[str, float, List[List[int]], str]:
    normalized = _resize_frame(frame)
    frame_height, frame_width = normalized.shape[:2]
    roi_left, roi_top, roi_right, roi_bottom = _center_roi_bounds(frame_width, frame_height)
    roi = normalized[roi_top:roi_bottom, roi_left:roi_right]

    grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    equalized = clahe.apply(grayscale)
    filtered = cv2.bilateralFilter(equalized, 9, 75, 75)
    thresholded = cv2.adaptiveThreshold(
        filtered,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        101,
        2,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    reinforced = cv2.dilate(thresholded, kernel, iterations=1)

    candidates = _extract_dot_candidates(reinforced, filtered)
    text, confidence, roi_boxes = _decode_braille(candidates)
    mapped_boxes = [[x + roi_left, y + roi_top, w, h] for x, y, w, h in roi_boxes]
    debug_image = _encode_debug_image(reinforced)
    return text, confidence, mapped_boxes, debug_image


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/process-frame")
async def process_frame(file: UploadFile = File(...)) -> Dict[str, object]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Received empty frame.")

    frame = _decode_image(raw)
    text, confidence, boxes, debug_image = process_braille_frame(frame)
    return {
        "text": text,
        "confidence": confidence,
        "boxes": boxes,
        "debug_image": debug_image,
    }
