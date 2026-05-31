from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="BrailleVision API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ROI_TOP_RATIO = 0.18
ROI_BOTTOM_RATIO = 0.84
ROI_LEFT_RATIO = 0.12
ROI_RIGHT_RATIO = 0.88


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


def _decode_image(raw_bytes: bytes) -> np.ndarray:
    array = np.frombuffer(raw_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise HTTPException(status_code=400, detail="Unable to decode image frame.")
    return frame


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def _roi_bounds(width: int, height: int) -> Tuple[int, int, int, int]:
    left = int(width * ROI_LEFT_RATIO)
    right = int(width * ROI_RIGHT_RATIO)
    top = int(height * ROI_TOP_RATIO)
    bottom = int(height * ROI_BOTTOM_RATIO)
    return left, top, right, bottom


def _cluster_positions(values: Iterable[float], tolerance: float) -> List[float]:
    ordered = sorted(value for value in values if value >= 0)
    if not ordered:
        return []

    clusters: List[List[float]] = [[ordered[0]]]
    for value in ordered[1:]:
        if abs(value - median(clusters[-1])) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [float(median(cluster)) for cluster in clusters]


def _estimate_spacing(levels: Sequence[float], fallback: float) -> float:
    if len(levels) < 2:
        return fallback
    gaps = [levels[index + 1] - levels[index] for index in range(len(levels) - 1)]
    valid_gaps = [gap for gap in gaps if gap > 0]
    if not valid_gaps:
        return fallback
    return float(median(valid_gaps))


def _hamming_distance(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(int(left != right) for left, right in zip(a, b))


def _extract_dot_candidates(binary: np.ndarray, grayscale: np.ndarray) -> List[DotCandidate]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[DotCandidate] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 15 or area > 240:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        if circularity < 0.75:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            continue

        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 0.8 or aspect_ratio > 1.2:
            continue

        contour_mask = np.zeros(grayscale.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        contour_pixels = grayscale[contour_mask == 255]
        if contour_pixels.size == 0:
            continue

        ring_mask = np.zeros(grayscale.shape, dtype=np.uint8)
        cv2.circle(
            ring_mask,
            (int(x + w / 2), int(y + h / 2)),
            int(max(w, h) * 1.4),
            255,
            thickness=-1,
        )
        ring_mask = cv2.subtract(ring_mask, contour_mask)
        ring_pixels = grayscale[ring_mask == 255]
        local_background = float(np.mean(ring_pixels)) if ring_pixels.size else float(np.mean(grayscale))
        local_foreground = float(np.mean(contour_pixels))
        contrast = max(0.0, local_background - local_foreground)

        candidates.append(
            DotCandidate(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=float(area),
                circularity=circularity,
                contrast=contrast,
            )
        )

    candidates.sort(key=lambda dot: (dot.y, dot.x))
    return candidates


def _intersects(dot: DotCandidate, quadrant: Tuple[float, float, float, float]) -> bool:
    qx1, qy1, qx2, qy2 = quadrant
    dx1 = float(dot.x)
    dy1 = float(dot.y)
    dx2 = float(dot.x + dot.w)
    dy2 = float(dot.y + dot.h)
    return dx1 < qx2 and dx2 > qx1 and dy1 < qy2 and dy2 > qy1


def _cluster_cells(candidates: List[DotCandidate], dot_diameter: float) -> List[List[DotCandidate]]:
    if not candidates:
        return []

    horizontal_threshold = max(dot_diameter * 2.5, 18.0)
    vertical_threshold = max(dot_diameter * 3.5, 24.0)
    visited = [False] * len(candidates)
    clusters: List[List[DotCandidate]] = []

    for start_index in range(len(candidates)):
        if visited[start_index]:
            continue

        queue = [start_index]
        visited[start_index] = True
        component: List[DotCandidate] = []

        while queue:
            index = queue.pop()
            current = candidates[index]
            component.append(current)
            current_x, current_y = current.center

            for neighbor_index, neighbor in enumerate(candidates):
                if visited[neighbor_index]:
                    continue
                neighbor_x, neighbor_y = neighbor.center
                if (
                    abs(neighbor_x - current_x) <= horizontal_threshold
                    and abs(neighbor_y - current_y) <= vertical_threshold
                ):
                    visited[neighbor_index] = True
                    queue.append(neighbor_index)

        clusters.append(sorted(component, key=lambda dot: (dot.y, dot.x)))

    return clusters


def _decode_cluster_pattern(
    dots: List[DotCandidate],
    dot_diameter: float,
) -> Tuple[Tuple[int, int, int, int, int, int], List[int], str]:
    anchor_dot = min(dots, key=lambda dot: (dot.y, dot.x))
    anchor_x, anchor_y = anchor_dot.center
    row_step = 2.5 * dot_diameter
    col_step = 2.5 * dot_diameter
    match_radius = 1.2 * dot_diameter

    theoretical_points = [
        (anchor_x, anchor_y),
        (anchor_x, anchor_y + row_step),
        (anchor_x, anchor_y + (2.0 * row_step)),
        (anchor_x + col_step, anchor_y),
        (anchor_x + col_step, anchor_y + row_step),
        (anchor_x + col_step, anchor_y + (2.0 * row_step)),
    ]

    pattern = [0, 0, 0, 0, 0, 0]
    for index, (target_x, target_y) in enumerate(theoretical_points):
        for dot in dots:
            center_x, center_y = dot.center
            if float(np.hypot(center_x - target_x, center_y - target_y)) <= match_radius:
                pattern[index] = 1
                break

    pattern_tuple = tuple(pattern)
    translation = GRADE_1_MAP.get(pattern_tuple, "")
    if not translation:
        best_pattern = min(
            GRADE_1_MAP.keys(),
            key=lambda candidate: _hamming_distance(pattern_tuple, candidate),
        )
        pattern_tuple = best_pattern
        translation = GRADE_1_MAP[best_pattern]

    min_x = min(dot.x for dot in dots)
    min_y = min(dot.y for dot in dots)
    max_x = max(dot.x + dot.w for dot in dots)
    max_y = max(dot.y + dot.h for dot in dots)
    anchor_x1 = min(min_x, anchor_x - match_radius)
    anchor_y1 = min(min_y, anchor_y - match_radius)
    anchor_x2 = max(max_x, anchor_x + col_step + match_radius)
    anchor_y2 = max(max_y, anchor_y + (2.0 * row_step) + match_radius)

    box = [
        max(0, int(round(anchor_x1))),
        max(0, int(round(anchor_y1))),
        max(1, int(round(anchor_x2 - anchor_x1))),
        max(1, int(round(anchor_y2 - anchor_y1))),
    ]
    return pattern_tuple, box, translation


def _build_cells(candidates: List[DotCandidate]) -> Tuple[str, float, List[List[int]]]:
    if len(candidates) < 1:
        return "", 0.0, []

    dot_diameter = float(np.mean([dot.w for dot in candidates]))
    cell_clusters = _cluster_cells(candidates, dot_diameter)
    if not cell_clusters:
        return "", 0.0, []

    line_threshold = max(dot_diameter * 4.0, 28.0)
    sorted_clusters = sorted(
        cell_clusters,
        key=lambda cluster: (
            min(dot.center[1] for dot in cluster),
            min(dot.center[0] for dot in cluster),
        ),
    )

    line_groups: List[List[List[DotCandidate]]] = []
    for cluster in sorted_clusters:
        cluster_top = min(dot.center[1] for dot in cluster)
        if not line_groups:
            line_groups.append([cluster])
            continue

        reference_top = median(
            min(dot.center[1] for dot in member)
            for member in line_groups[-1]
        )
        if abs(cluster_top - reference_top) <= line_threshold:
            line_groups[-1].append(cluster)
        else:
            line_groups.append([cluster])

    decoded_lines: List[str] = []
    confidences: List[float] = []
    ordered_boxes: List[List[int]] = []

    for line_clusters in line_groups:
        line_clusters.sort(key=lambda cluster: min(dot.center[0] for dot in cluster))
        row_text = ""
        previous_right_edge: float | None = None

        for cluster in line_clusters:
            if len(cluster) == 1:
                lone_dot = cluster[0]
                if lone_dot.circularity < 0.88 or lone_dot.contrast < 18.0:
                    previous_right_edge = max(lone_dot.x + lone_dot.w, previous_right_edge or 0.0)
                    continue

            pattern, anchor_box, translated = _decode_cluster_pattern(cluster, dot_diameter)

            cluster_left = min(dot.x for dot in cluster)
            cluster_right = max(dot.x + dot.w for dot in cluster)
            if previous_right_edge is not None:
                if cluster_left - previous_right_edge > dot_diameter * 2.8:
                    row_text += " "
            previous_right_edge = cluster_right

            row_text += translated
            ordered_boxes.append(anchor_box)

            occupancy_score = sum(pattern) / 6.0
            exact_match_score = 1.0 if pattern in GRADE_1_MAP else 0.65
            geometry_score = float(np.mean([dot.circularity for dot in cluster]))
            contrast_score = min(1.0, float(np.mean([dot.contrast for dot in cluster])) / 32.0)
            cluster_confidence = min(
                1.0,
                0.15
                + occupancy_score * 0.3
                + exact_match_score * 0.2
                + geometry_score * 0.2
                + contrast_score * 0.15,
            )
            confidences.append(cluster_confidence)

        cleaned = row_text.strip()
        if cleaned:
            decoded_lines.append(cleaned)

    text = "\n".join(decoded_lines).strip()
    if not text:
        return "", 0.0, []

    confidence = round(float(np.mean(confidences)), 3) if confidences else 0.0
    return text, confidence, ordered_boxes


def process_braille_frame(frame: np.ndarray) -> Tuple[str, float, List[List[int]]]:
    normalized = _resize_frame(frame)
    frame_height, frame_width = normalized.shape[:2]
    roi_left, roi_top, roi_right, roi_bottom = _roi_bounds(frame_width, frame_height)
    roi = normalized[roi_top:roi_bottom, roi_left:roi_right]

    grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    normalized_contrast = clahe.apply(blurred)
    thresholded = cv2.adaptiveThreshold(
        normalized_contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        4,
    )

    candidates = _extract_dot_candidates(thresholded, normalized_contrast)
    text, confidence, roi_boxes = _build_cells(candidates)

    mapped_boxes = [
        [x + roi_left, y + roi_top, w, h]
        for x, y, w, h in roi_boxes
    ]
    return text, confidence, mapped_boxes


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/process-frame")
async def process_frame(file: UploadFile = File(...)) -> Dict[str, object]:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Received empty frame.")

    frame = _decode_image(contents)
    text, confidence, boxes = process_braille_frame(frame)
    return {
        "text": text,
        "confidence": confidence,
        "boxes": boxes,
    }
