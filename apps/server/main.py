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


FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROI_TOP_RATIO = 0.20
ROI_BOTTOM_RATIO = 0.80
ROI_LEFT_RATIO = 0.15
ROI_RIGHT_RATIO = 0.85


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


def _extract_dot_candidates(binary: np.ndarray) -> List[DotCandidate]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[DotCandidate] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5 or area > 150:
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

        candidates.append(
            DotCandidate(
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
                area=float(area),
                circularity=circularity,
            )
        )

    candidates.sort(key=lambda dot: (dot.y, dot.x))
    return candidates


def _build_cells(candidates: List[DotCandidate]) -> Tuple[str, float, List[List[int]]]:
    if len(candidates) < 2:
        return "", 0.0, []

    avg_width = float(np.mean([dot.w for dot in candidates]))
    avg_height = float(np.mean([dot.h for dot in candidates]))
    col_tolerance = max(avg_width * 0.8, 6.0)
    row_tolerance = max(avg_height * 0.8, 6.0)

    x_levels = _cluster_positions((dot.center[0] for dot in candidates), col_tolerance)
    y_levels = _cluster_positions((dot.center[1] for dot in candidates), row_tolerance)
    if len(x_levels) < 2 or len(y_levels) < 3:
        return "", 0.0, []

    horizontal_spacing = _estimate_spacing(x_levels, fallback=max(avg_width * 1.8, 12.0))
    vertical_spacing = _estimate_spacing(y_levels, fallback=max(avg_height * 1.7, 12.0))
    cell_width = max(horizontal_spacing * 2.2, avg_width * 2.5)
    cell_height = max(vertical_spacing * 3.2, avg_height * 3.5)

    grouped_rows: List[List[float]] = []
    for level in y_levels:
        if not grouped_rows or abs(level - grouped_rows[-1][0]) > vertical_spacing * 2.0:
            grouped_rows.append([level])
        else:
            grouped_rows[-1].append(level)
    row_bases = [float(min(group)) for group in grouped_rows]

    grouped_cols: List[List[float]] = []
    for level in x_levels:
        if not grouped_cols or abs(level - grouped_cols[-1][0]) > horizontal_spacing * 1.6:
            grouped_cols.append([level])
        else:
            grouped_cols[-1].append(level)
    col_bases = [float(min(group)) for group in grouped_cols]

    cells: Dict[Tuple[int, int], List[DotCandidate]] = {}
    cell_boxes: Dict[Tuple[int, int], List[int]] = {}

    for dot in candidates:
        cx, cy = dot.center
        row_idx = min(range(len(row_bases)), key=lambda index: abs(cy - row_bases[index]))
        col_idx = min(range(len(col_bases)), key=lambda index: abs(cx - col_bases[index]))
        key = (row_idx, col_idx)
        cells.setdefault(key, []).append(dot)

        x1 = dot.x
        y1 = dot.y
        x2 = dot.x + dot.w
        y2 = dot.y + dot.h
        if key not in cell_boxes:
            cell_boxes[key] = [x1, y1, x2, y2]
        else:
            cell_boxes[key][0] = min(cell_boxes[key][0], x1)
            cell_boxes[key][1] = min(cell_boxes[key][1], y1)
            cell_boxes[key][2] = max(cell_boxes[key][2], x2)
            cell_boxes[key][3] = max(cell_boxes[key][3], y2)

    decoded_lines: List[str] = []
    confidences: List[float] = []
    ordered_boxes: List[List[int]] = []

    for row_idx in sorted({key[0] for key in cells.keys()}):
        row_keys = sorted((key for key in cells.keys() if key[0] == row_idx), key=lambda item: item[1])
        if not row_keys:
            continue

        row_text = ""
        previous_col: int | None = None

        for key in row_keys:
            current_col = key[1]
            if previous_col is not None and current_col - previous_col > 1:
                row_text += " "
            previous_col = current_col

            dots = cells[key]
            base_x = col_bases[current_col]
            base_y = row_bases[row_idx]
            x_targets = [base_x, base_x + horizontal_spacing]
            y_targets = [base_y, base_y + vertical_spacing, base_y + 2 * vertical_spacing]

            pattern = [0, 0, 0, 0, 0, 0]
            matched_positions: set[Tuple[int, int]] = set()

            for dot in dots:
                cx, cy = dot.center
                col_slot = min(range(2), key=lambda index: abs(cx - x_targets[index]))
                row_slot = min(range(3), key=lambda index: abs(cy - y_targets[index]))
                pattern_index = row_slot + col_slot * 3
                pattern[pattern_index] = 1
                matched_positions.add((col_slot, row_slot))

            pattern_tuple = tuple(pattern)
            translated = GRADE_1_MAP.get(pattern_tuple, "?")
            row_text += translated

            occupancy_score = len(matched_positions) / 6.0
            dictionary_score = 1.0 if translated != "?" else 0.35
            geometry_score = float(np.mean([dot.circularity for dot in dots])) if dots else 0.0
            cell_confidence = min(
                1.0,
                0.2 + occupancy_score * 0.35 + dictionary_score * 0.25 + geometry_score * 0.2,
            )
            confidences.append(cell_confidence)

            x1, y1, x2, y2 = cell_boxes[key]
            padded_x1 = max(0, int(round(min(x1, base_x - avg_width * 0.6))))
            padded_y1 = max(0, int(round(min(y1, base_y - avg_height * 0.6))))
            padded_x2 = int(round(max(x2, base_x + cell_width)))
            padded_y2 = int(round(max(y2, base_y + cell_height)))
            ordered_boxes.append(
                [padded_x1, padded_y1, max(1, padded_x2 - padded_x1), max(1, padded_y2 - padded_y1)]
            )

        if row_text.strip():
            decoded_lines.append(row_text.strip())

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
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        10,
    )

    candidates = _extract_dot_candidates(thresholded)
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
