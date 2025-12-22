#!/usr/bin/env python3
"""
visualize_legal_paths.py

Visualize all legal path segments within a cropped region from a town node file.
Legal paths follow yaw requirements and don't make crazy turns.

NEW (LLM-friendly):
- Writes a single combined “prompt file” that contains all candidate paths in a compact,
  structured "ideal path representation" (no long natural-language descriptions).
- This file is meant to be copy-pasted directly into a local LLM prompt.

Usage:
  python visualize_legal_paths.py --nodes town_nodes/Town05.json \
    --crop -40 40 -40 40 \
    --max-yaw-diff 60 \
    --min-path-length 20 \
    --viz \
    --out-prompt legal_paths_prompt.txt

Arguments:
  --nodes: Path to the town node JSON file
  --crop: Bounding box as xmin xmax ymin ymax
  --max-yaw-diff: Maximum yaw difference between connected segments (degrees)
  --min-path-length: Minimum total path length to display (meters)
  --max-paths: Maximum number of paths to generate (default: 100)
  --viz: Display the visualization
  --out: Output image file path (default: legal_paths_viz.png)
  --out-prompt: Output prompt-ready text file (default: legal_paths_prompt.txt)
  --out-json-detailed: Output aggregated JSON with signatures (optional; machine-readable)
  --individual / --individual-dir: Optional per-path visualization
"""

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except Exception:
    plt = None


# ============================================================================
# Utility Functions
# ============================================================================

def wrap180(deg: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap degrees into [-180, 180)."""
    arr = np.asarray(deg)
    wrapped = ((arr + 180.0) % 360.0) - 180.0
    return float(wrapped) if arr.shape == () else wrapped


def ang_diff_deg(a: float, b: float) -> float:
    """Absolute wrapped difference in degrees."""
    return float(abs(wrap180(a - b)))


def heading_deg_from_vec(v: np.ndarray) -> float:
    """Compute heading in degrees from a 2D vector."""
    return float(math.degrees(math.atan2(v[1], v[0])))


def unit_from_yaw_deg(yaw_deg: float) -> np.ndarray:
    """Convert yaw in degrees to unit vector."""
    r = math.radians(float(wrap180(yaw_deg)))
    return np.array([math.cos(r), math.sin(r)], dtype=float)


def cumulative_dist(points_xy: np.ndarray) -> np.ndarray:
    """Compute cumulative arc-length along polyline."""
    if len(points_xy) < 2:
        return np.array([0.0])
    seg = np.linalg.norm(points_xy[1:] - points_xy[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def sanitize(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "-")
         .replace("\\", "-")
         .replace(":", "-")
         .replace("|", "-")
    )


# ============================================================================
# World-frame heading helpers (recommended for LLM selection)
# ============================================================================

def classify_turn_world(in_heading: float, out_heading: float) -> str:
    """
    Classify turn in canonical world frame used here:
    - Heading computed from atan2(y, x)
    - X increases → West, X decreases → East (mirrored X)

    Mirroring X flips handedness; thus left/right are inverted compared to the
    standard math frame. We flip the sign test accordingly.
    """
    d = wrap180(out_heading - in_heading)
    ad = abs(d)
    if ad <= 35:
        return "straight"
    # Note the swapped semantics due to mirrored X axis
    if 35 < d < 145:
        return "right"
    if -145 < d < -35:
        return "left"
    return "uturn"


def heading_to_cardinal4_world(h: float) -> str:
    """
    4-way cardinal (E/N/W/S) in canonical world frame where:
    - X+ is West, X- is East. Therefore:
      - Heading ~ 0° (along +X) → "W"
      - Heading ~ ±180° (along -X) → "E"
      - Heading ~ 90° → "N"; ~ -90° → "S"
    """
    hd = wrap180(float(h))
    snapped = int(round(hd / 90.0)) * 90
    snapped = int(wrap180(snapped))
    if -45 <= snapped < 45:
        return "W"
    if 45 <= snapped < 135:
        return "N"
    if snapped >= 135 or snapped < -135:
        return "E"
    return "S"


def bound_word(card4: str) -> str:
    return {"E": "eastbound", "W": "westbound", "N": "northbound", "S": "southbound"}.get(card4, "forward")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LaneSegment:
    """Represents a continuous lane segment with geometry and metadata."""
    seg_id: int
    road_id: int
    lane_id: int
    section_id: int
    points: np.ndarray      # (N, 2) x,y coordinates
    yaws: np.ndarray        # (N,) heading at each point
    orig_idx: np.ndarray    # (N,) indices into original node file

    def bbox(self) -> Tuple[float, float, float, float]:
        """Return bounding box (xmin, xmax, ymin, ymax)."""
        mn = self.points.min(axis=0)
        mx = self.points.max(axis=0)
        return float(mn[0]), float(mx[0]), float(mn[1]), float(mx[1])

    def heading_at_start(self, k: int = 6) -> float:
        """Compute heading at start using first k+1 points."""
        k = min(k, len(self.points) - 1)
        if k == 0:
            return float(self.yaws[0])
        v = self.points[k] - self.points[0]
        n = np.linalg.norm(v)
        if n < 1e-6:
            return float(self.yaws[0])
        return heading_deg_from_vec(v)

    def heading_at_end(self, k: int = 6) -> float:
        """Compute heading at end using last k+1 points."""
        k = min(k, len(self.points) - 1)
        if k == 0:
            return float(self.yaws[-1])
        v = self.points[-1] - self.points[-(k+1)]
        n = np.linalg.norm(v)
        if n < 1e-6:
            return float(self.yaws[-1])
        return heading_deg_from_vec(v)

    def length(self) -> float:
        """Total arc length of the segment."""
        return float(cumulative_dist(self.points)[-1])


@dataclass
class CropBox:
    """Rectangular region for filtering segments."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, p: np.ndarray) -> bool:
        """Check if point is inside the crop box."""
        return (self.xmin <= float(p[0]) <= self.xmax and
                self.ymin <= float(p[1]) <= self.ymax)

    def intersects_bbox(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if this crop box intersects with another bounding box."""
        x0, x1, y0, y1 = bbox
        return not (x1 < self.xmin or x0 > self.xmax or
                    y1 < self.ymin or y0 > self.ymax)


@dataclass
class LegalPath:
    """A legal multi-segment path through the road network."""
    segments: List[LaneSegment]
    total_length: float


# ============================================================================
# Segment Loading and Processing
# ============================================================================

def orient_polyline(points_xy: np.ndarray, yaws_deg: np.ndarray,
                    orig_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure polyline direction agrees with yaw direction.
    If average dot(direction_vec, yaw_vec) < 0, reverse the polyline.
    """
    pts = np.asarray(points_xy, dtype=float)
    yaws = wrap180(np.asarray(yaws_deg, dtype=float))
    idxs = np.asarray(orig_idx, dtype=int)

    if len(pts) < 2:
        return pts, yaws, idxs

    vecs = pts[1:] - pts[:-1]
    norms = np.linalg.norm(vecs, axis=1) + 1e-9
    dir_vecs = vecs / norms[:, None]

    yaw_vecs = np.vstack([unit_from_yaw_deg(y) for y in yaws[:-1]])
    dots = np.sum(dir_vecs * yaw_vecs, axis=1)

    if float(np.nanmean(dots)) < 0.0:
        pts = pts[::-1].copy()
        yaws = yaws[::-1].copy()
        idxs = idxs[::-1].copy()

    return pts, yaws, idxs


def split_by_gaps(idxs_sorted: np.ndarray, pts: np.ndarray, yaws: np.ndarray,
                  gap_m: float = 6.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Split a polyline into continuous chunks at large gaps."""
    if len(pts) < 2:
        return [(idxs_sorted, pts, yaws)] if len(pts) > 0 else []

    jumps = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    cuts = [0]
    for i, d in enumerate(jumps):
        if float(d) > gap_m:
            cuts.append(i + 1)
    cuts.append(len(pts))

    chunks = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a >= 2:
            chunks.append((idxs_sorted[a:b], pts[a:b], yaws[a:b]))

    return chunks


def load_nodes(path: str) -> Dict[str, Any]:
    """Load town nodes JSON file."""
    with open(path, "r") as f:
        data = json.load(f)
    if "payload" not in data:
        raise ValueError(f"{path} does not contain a top-level 'payload' field")
    return data


def build_segments(data: Dict[str, Any], min_points: int = 6) -> List[LaneSegment]:
    """
    Build lane segments from node data.
    Groups waypoints by (road_id, lane_id, section_id) and orients them correctly.
    """
    payload = data["payload"]
    required_keys = ["x", "y", "yaw", "road_id", "lane_id", "section_id"]
    for k in required_keys:
        if k not in payload:
            raise ValueError(f"payload missing required key '{k}'")

    x = np.asarray(payload["x"], dtype=float)
    y = np.asarray(payload["y"], dtype=float)
    yaw = np.asarray(payload["yaw"], dtype=float)
    road_id = np.asarray(payload["road_id"], dtype=int)
    lane_id = np.asarray(payload["lane_id"], dtype=int)
    section_id = np.asarray(payload["section_id"], dtype=int)

    grouped: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in range(len(x)):
        grouped[(int(road_id[i]), int(lane_id[i]), int(section_id[i]))].append(i)

    segments: List[LaneSegment] = []
    seg_id = 0

    for (rid, lid, sid), idxs in grouped.items():
        idxs_sorted = np.asarray(sorted(idxs), dtype=int)
        pts = np.vstack([x[idxs_sorted], y[idxs_sorted]]).T
        yaws_data = yaw[idxs_sorted]

        for idxs_chunk, pts_chunk, yaws_chunk in split_by_gaps(idxs_sorted, pts, yaws_data):
            pts_o, yaws_o, idxs_o = orient_polyline(pts_chunk, yaws_chunk, idxs_chunk)
            if len(pts_o) < min_points:
                continue

            segments.append(
                LaneSegment(
                    seg_id=seg_id,
                    road_id=int(rid),
                    lane_id=int(lid),
                    section_id=int(sid),
                    points=pts_o,
                    yaws=wrap180(yaws_o),
                    orig_idx=idxs_o,
                )
            )
            seg_id += 1

    return segments


def crop_segments(segments: List[LaneSegment], crop: CropBox) -> List[LaneSegment]:
    """Filter segments that intersect with the crop box."""
    return [s for s in segments if crop.intersects_bbox(s.bbox())]


# ============================================================================
# Connectivity and Path Generation
# ============================================================================

def build_connectivity(segments: List[LaneSegment],
                       connect_radius_m: float = 6.0,
                       connect_yaw_tol_deg: float = 60.0) -> List[List[int]]:
    """
    Build segment-to-segment connectivity graph.
    Two segments are connected if:
    1. End of segment A is close to start of segment B (within connect_radius_m)
    2. Heading at end of A aligns with heading at start of B (within connect_yaw_tol_deg)
    """
    n = len(segments)
    adj: List[List[int]] = [[] for _ in range(n)]
    if n == 0:
        return adj

    starts = np.vstack([seg.points[0] for seg in segments])
    tree = cKDTree(starts)

    for i, seg_a in enumerate(segments):
        end_pt = seg_a.points[-1]
        end_heading = seg_a.heading_at_end()

        candidates = tree.query_ball_point(end_pt, r=connect_radius_m)
        for j in candidates:
            if i == j:
                continue
            seg_b = segments[j]
            start_heading = seg_b.heading_at_start()
            if ang_diff_deg(end_heading, start_heading) <= connect_yaw_tol_deg:
                adj[i].append(j)

    return adj


def identify_boundary_segments(segments: List[LaneSegment],
                              crop: CropBox) -> Tuple[List[int], List[int]]:
    """
    Identify segments that cross the crop boundary.

    Entry segments: start outside, enter inside
    Exit segments: start inside, exit outside
    """
    entry_segments = []
    exit_segments = []

    for i, seg in enumerate(segments):
        start_inside = crop.contains(seg.points[0])
        end_inside = crop.contains(seg.points[-1])

        if not start_inside and end_inside:
            entry_segments.append(i)
        elif not start_inside and not end_inside:
            for pt in seg.points[1:-1]:
                if crop.contains(pt):
                    entry_segments.append(i)
                    break

        if start_inside and not end_inside:
            exit_segments.append(i)
        elif not start_inside and not end_inside:
            if i not in entry_segments:
                for pt in seg.points[1:-1]:
                    if crop.contains(pt):
                        exit_segments.append(i)
                        break

    return entry_segments, exit_segments


def generate_legal_paths(segments: List[LaneSegment],
                        adj: List[List[int]],
                        crop: CropBox,
                        min_path_length: float = 20.0,
                        max_paths: int = 100,
                        max_depth: int = 10,
                        allow_within_region_fallback: bool = True) -> List[LegalPath]:
    """
    Generate legal paths that go from outside the crop area to outside.
    """
    legal_paths: List[LegalPath] = []

    entry_segments, exit_segments = identify_boundary_segments(segments, crop)
    print(f"[INFO] Found {len(entry_segments)} entry segments and {len(exit_segments)} exit segments")

    if len(entry_segments) == 0 or len(exit_segments) == 0:
        print("[WARNING] No entry or exit segments found. Paths must cross crop boundary.")
        if not allow_within_region_fallback:
            print("[INFO] Returning 0 legal paths for this crop (requires boundary-crossing paths).")
            return []
        print("[INFO] Falling back to any paths within the region...")
        entry_segments = list(range(len(segments)))
        exit_segments = list(range(len(segments)))

    exit_set = set(exit_segments)

    def dfs(current_idx: int, path: List[int], total_length: float, depth: int):
        if len(legal_paths) >= max_paths:
            return

        if current_idx in exit_set and len(path) >= 2:
            if total_length >= min_path_length:
                path_segments = [segments[i] for i in path]
                legal_paths.append(LegalPath(path_segments, total_length))
                return

        if depth >= max_depth:
            return

        for next_idx in adj[current_idx]:
            if next_idx in path:
                continue
            next_seg = segments[next_idx]
            new_length = total_length + next_seg.length()
            dfs(next_idx, path + [next_idx], new_length, depth + 1)

    for entry_idx in entry_segments:
        if len(legal_paths) >= max_paths:
            break
        dfs(entry_idx, [entry_idx], segments[entry_idx].length(), 1)

    return legal_paths


# ============================================================================
# Ideal Path Representation (structured “signature”)
# ============================================================================

def _pt2(p: np.ndarray) -> Dict[str, float]:
    return {"x": float(p[0]), "y": float(p[1])}


# --------------------------
# EXTRA LOGGING FOR JSON ONLY
# --------------------------

def _sample_polyline(points_xy: np.ndarray, max_points: int = 10) -> List[Dict[str, float]]:
    """
    Small, deterministic polyline sample for logging/debug.
    This is NOT used for any connectivity/path logic.
    """
    pts = np.asarray(points_xy, dtype=float)
    n = len(pts)
    if n == 0:
        return []
    if n <= max_points:
        return [{"x": float(p[0]), "y": float(p[1])} for p in pts]
    idxs = np.linspace(0, n - 1, num=max_points, dtype=int)
    return [{"x": float(pts[i, 0]), "y": float(pts[i, 1])} for i in idxs]


def build_segments_detailed_for_path(path: LegalPath, polyline_sample_n: int = 10) -> List[Dict[str, Any]]:
    """
    Extra per-segment logging payload for --out-json-detailed ONLY.
    No effect on any functional behavior; purely additional metadata export.
    """
    out: List[Dict[str, Any]] = []
    for seg in path.segments:
        start_pt = seg.points[0]
        end_pt = seg.points[-1]

        hs = float(wrap180(seg.heading_at_start()))
        he = float(wrap180(seg.heading_at_end()))
        cs = heading_to_cardinal4_world(hs)
        ce = heading_to_cardinal4_world(he)

        bb = seg.bbox()
        out.append({
            "seg_id": int(seg.seg_id),
            "road_id": int(seg.road_id),
            "section_id": int(seg.section_id),
            "lane_id": int(seg.lane_id),
            "length_m": float(seg.length()),
            "bbox": {
                "xmin": float(bb[0]),
                "xmax": float(bb[1]),
                "ymin": float(bb[2]),
                "ymax": float(bb[3]),
            },
            "start": {
                "point": _pt2(start_pt),
                "heading_deg": hs,
                "cardinal4": cs,
                "bound": bound_word(cs),
                "orig_idx": int(seg.orig_idx[0]) if len(seg.orig_idx) > 0 else None,
            },
            "end": {
                "point": _pt2(end_pt),
                "heading_deg": he,
                "cardinal4": ce,
                "bound": bound_word(ce),
                "orig_idx": int(seg.orig_idx[-1]) if len(seg.orig_idx) > 0 else None,
            },
            "polyline_sample": _sample_polyline(seg.points, max_points=polyline_sample_n),
        })
    return out


def build_path_signature(path: LegalPath) -> Dict[str, Any]:
    """
    Compact, order-explicit representation that an LLM can match reliably.
    Uses WORLD-FRAME turn classification (recommended).
    """
    segs = path.segments
    assert len(segs) > 0

    entry_h = float(wrap180(segs[0].heading_at_start()))
    exit_h = float(wrap180(segs[-1].heading_at_end()))
    entry_card4 = heading_to_cardinal4_world(entry_h)
    exit_card4 = heading_to_cardinal4_world(exit_h)

    # Ordered maneuvers between segments (length = nseg-1)
    maneuvers_between = []
    for i in range(len(segs) - 1):
        in_h = segs[i].heading_at_end()
        out_h = segs[i + 1].heading_at_start()
        maneuvers_between.append(classify_turn_world(in_h, out_h))

    # Roads / sections / lanes sequences
    roads = [int(s.road_id) for s in segs]
    sections = [int(s.section_id) for s in segs]
    lanes = [int(s.lane_id) for s in segs]
    seg_ids = [int(s.seg_id) for s in segs]

    # Start/end points
    start_pt = segs[0].points[0]
    end_pt = segs[-1].points[-1]

    return {
        "entry": {
            "point": _pt2(start_pt),
            "heading_deg": entry_h,
            "cardinal4": entry_card4,
            "bound": bound_word(entry_card4),
            "road_id": int(segs[0].road_id),
            "section_id": int(segs[0].section_id),
            "lane_id": int(segs[0].lane_id),
        },
        "exit": {
            "point": _pt2(end_pt),
            "heading_deg": exit_h,
            "cardinal4": exit_card4,
            "bound": bound_word(exit_card4),
            "road_id": int(segs[-1].road_id),
            "section_id": int(segs[-1].section_id),
            "lane_id": int(segs[-1].lane_id),
        },
        "length_m": float(path.total_length),
        "num_segments": int(len(segs)),
        "segment_ids": seg_ids,
        "roads": roads,
        "sections": sections,
        "lanes": lanes,
        "maneuvers_between": maneuvers_between,  # order-explicit
        # Helpful redundancy: coarse path “turn” from entry->exit (not perfect, but quick filter)
        "entry_to_exit_turn": classify_turn_world(entry_h, exit_h),
    }


def make_path_name(path_idx: int, sig: Dict[str, Any]) -> str:
    """Stable-ish name for humans + logs."""
    ent = sig["entry"]
    ex = sig["exit"]
    man = sig["entry_to_exit_turn"]
    nseg = sig["num_segments"]
    length = sig["length_m"]
    return (
        f"path_{path_idx + 1:03d}"
        f"__man={sanitize(man)}"
        f"__entry={ent['cardinal4']}({int(round(ent['heading_deg']))}deg)"
        f"__exit={ex['cardinal4']}({int(round(ex['heading_deg']))}deg)"
        f"__len={length:.1f}m__n={nseg}segs"
    )


def save_prompt_file(
    out_path: str,
    crop: CropBox,
    nodes_path: str,
    params: Dict[str, Any],
    paths_named: List[Dict[str, Any]],
) -> None:
    """
    Writes a prompt-ready TEXT file (copy-paste into your LLM).

    OPTIMIZED FOR PATH PICKING:
    - The prompt includes a COMPACT candidate list that is easy for an LLM to scan.
    - It intentionally omits heavy per-segment fields (e.g., segments_detailed) because those
      often degrade selection accuracy by adding noise and increasing context length.
    - If you need segment-level placement (e.g., "immediately after the turn"), use the
      machine-readable JSON output (--out-json-detailed) after the path is selected.

    The compact candidate schema is:
      {
        "name": "...",
        "entry": {"cardinal4": "N", "heading_deg": 90, "road_id": 9, "lane_id": 1},
        "exit":  {"cardinal4": "E", "heading_deg": 0,  "road_id": 10,"lane_id": 1},
        "entry_to_exit_turn": "right",
        "maneuvers_between": ["right","straight"],
        "num_segments": 3,
        "length_m": 85.1
      }
    """
    # Build a compact view of candidates for better LLM accuracy.
    compact_candidates: List[Dict[str, Any]] = []
    for c in paths_named:
        sig = c.get("signature", {})
        ent = sig.get("entry", {})
        ex = sig.get("exit", {})
        compact_candidates.append({
            "name": c.get("name", ""),
            "entry": {
                "cardinal4": ent.get("cardinal4", None),
                "heading_deg": ent.get("heading_deg", None),
                "road_id": ent.get("road_id", None),
                "lane_id": ent.get("lane_id", None),
            },
            "exit": {
                "cardinal4": ex.get("cardinal4", None),
                "heading_deg": ex.get("heading_deg", None),
                "road_id": ex.get("road_id", None),
                "lane_id": ex.get("lane_id", None),
            },
            "entry_to_exit_turn": sig.get("entry_to_exit_turn", None),
            "maneuvers_between": sig.get("maneuvers_between", None),
            "num_segments": sig.get("num_segments", None),
            "length_m": sig.get("length_m", None),
        })

    payload = {
        "context": {
            "nodes": nodes_path,
            "crop_region": {"xmin": crop.xmin, "xmax": crop.xmax, "ymin": crop.ymin, "ymax": crop.ymax},
            "parameters": params,
            "num_candidates": len(compact_candidates),
            "turn_frame": params.get("turn_frame", "WORLD_FRAME"),
        },
        "candidates": compact_candidates,
        "output_schema": {
            "vehicles": [
                {"vehicle": "Vehicle 1", "path_name": "path_###__...", "confidence": 0.0}
            ]
        },
    }

    instructions = (
        "You are given a list of candidate vehicle paths in a road network.\n"
        "Each candidate has a NAME plus a compact SIGNATURE describing its entry direction,\n"
        "exit direction, and overall maneuver.\n"
        "\n"
        "TASK:\n"
        "Given a user scenario description (vehicles and their intended motions), choose the\n"
        "best matching candidate path NAME for each moving vehicle.\n"
        "Ignore static or parked objects; only assign paths to moving vehicles.\n"
        "\n"
        "HOW TO INTERPRET THE DESCRIPTION (IMPORTANT):\n"
        "- Vehicles are described relative to roads, directions, and other vehicles.\n"
        "- Phrases like 'coming from X', 'approaching from X', or 'on the X road' constrain\n"
        "  the ENTRY direction of the path.\n"
        "- Phrases like 'onto X', 'to X', or 'leaves onto X' constrain the EXIT direction.\n"
        "- If a vehicle is described as coming from the same direction as another vehicle,\n"
        "  their chosen paths should have the same or compatible entry direction.\n"
        "- If a T-junction is mentioned, the 'main road' refers to the straight-through road\n"
        "  and the 'side road' refers to the terminating branch.\n"
        "\n"
        "PERPENDICULAR / CROSS-TRAFFIC INTERPRETATION (CRITICAL):\n"
        "- 'Perpendicular road to the RIGHT of Vehicle X's approach': if Vehicle X is westbound (W),\n"
        "  the road to their right is the southbound road (entry=S). If Vehicle X is eastbound (E),\n"
        "  the road to their right is the northbound road (entry=N). Apply 90-degree clockwise rotation.\n"
        "- 'Perpendicular road to the LEFT of Vehicle X's approach': apply 90-degree counter-clockwise.\n"
        "- 'Turns right onto Vehicle X's exit road': the vehicle's EXIT direction must match Vehicle X's\n"
        "  exit direction. If Vehicle X exits westbound (W), the turning vehicle must also exit W.\n"
        "- 'Turns left onto Vehicle X's exit road': same logic - EXIT directions must match.\n"
        "\n"
        "PATH SELECTION GUIDELINES:\n"
        "- First match the described source and destination semantics (where the vehicle comes from and goes to).\n"
        "- Then match the maneuver (left, right, straight).\n"
        "- Prefer paths whose entry and exit directions best align with the description,\n"
        "  even if multiple candidates share the same maneuver.\n"
        "- A path that reverses 'from' vs 'onto' semantics is a poor match and should only\n"
        "  be chosen if no better option exists.\n"
        "\n"
        "IMPORTANT RULES:\n"
        "1) Return JSON ONLY (no prose, no markdown, no code fences).\n"
        "2) path_name MUST be copied EXACTLY from one of the provided candidate name strings.\n"
        "3) If ambiguous, still pick the best match and include confidence in [0,1].\n"
        "\n"
        "CANDIDATE PATHS (JSON):\n"
    )

    with open(out_path, "w") as f:
        f.write(instructions)
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"[INFO] Prompt-ready file saved to: {out_path}")

    # Also write a compact prompt template for CONSTRAINT EXTRACTION (optional, for debugging / inspection).
    # This is NOT used programmatically by default; the pipeline may build its own constraints prompt.
    try:
        import os as _os
        _constraints_path = _os.path.join(_os.path.dirname(out_path), "constraints_prompt.txt")
        _constraints_instructions = (
            "You will read a short driving scene description.\n"
            "Extract constraints about MOVING vehicles only (Vehicle 1, Vehicle 2, ...).\n"
            "Do NOT choose path names here; only list constraints.\n"
            "\n"
            "Return JSON ONLY with this schema:\n"
            "{\n"
            "  \"vehicles\": [\n"
            "    {\n"
            "      \"vehicle\": \"Vehicle 1\",\n"
            "      \"maneuver\": \"left\" | \"right\" | \"straight\" | \"unknown\",\n"
            "      \"entry_road\": \"main\" | \"side\" | \"unknown\",\n"
            "      \"exit_road\": \"main\" | \"side\" | \"unknown\"\n"
            "    }\n"
            "  ],\n"
            "  \"constraints\": [\n"
            "    { \"type\": \"same_entry_as\" | \"opposite_entry_of\" | \"same_exit_as\" | \"opposite_exit_of\", \"a\": \"Vehicle X\", \"b\": \"Vehicle Y\" }\n"
            "  ]\n"
            "}\n"
            "\n"
            "Guidance:\n"
            "- If the description says \"coming from the same direction as Vehicle K\", output type=\"same_entry_as\".\n"
            "- If it says \"coming from the opposite direction\", output type=\"opposite_entry_of\".\n"
            "- If it mentions \"on the main road\" or \"on the side road\", set entry_road accordingly.\n"
            "- If it mentions \"onto the main road\" or \"onto the side road\", set exit_road accordingly.\n"
            "- Only include constraints that are explicitly stated.\n"
            "\n"
            "DESCRIPTION:\n"
        )
        with open(_constraints_path, "w", encoding="utf-8") as _f:
            _f.write(_constraints_instructions)
        print(f"[INFO] Constraint-extraction prompt template saved to: {_constraints_path}")
    except Exception as _e:
        print(f"[WARNING] Failed to write constraints_prompt.txt: {_e}")




def save_aggregated_signatures_json(
    out_path: str,
    crop: CropBox,
    nodes_path: str,
    params: Dict[str, Any],
    paths_named: List[Dict[str, Any]],
) -> None:
    """Optional: pure JSON version (no instructions), good for programmatic prompting."""
    payload = {
        "nodes": nodes_path,
        "crop_region": {"xmin": crop.xmin, "xmax": crop.xmax, "ymin": crop.ymin, "ymax": crop.ymax},
        "parameters": params,
        "num_candidates": len(paths_named),
        "candidates": paths_named,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Aggregated signatures JSON saved: {out_path}")


# ============================================================================
# Visualization
# ============================================================================

def visualize_legal_paths(segments: List[LaneSegment],
                         legal_paths: List[LegalPath],
                         crop: CropBox,
                         out_path: str):
    if plt is None:
        raise RuntimeError("matplotlib is not available for visualization")

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(crop.xmin - 5, crop.xmax + 5)
    ax.set_ylim(crop.ymin - 5, crop.ymax + 5)
    ax.invert_xaxis()
    ax.set_xlabel("X (meters)", fontsize=12)
    ax.set_ylabel("Y (meters)", fontsize=12)
    ax.set_title(f"Legal Path Segments (Total: {len(legal_paths)} paths)",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    rect = mpatches.Rectangle(
        (crop.xmin, crop.ymin),
        crop.xmax - crop.xmin,
        crop.ymax - crop.ymin,
        linewidth=2,
        edgecolor='red',
        facecolor='none',
        linestyle='--',
        label='Crop Region'
    )
    ax.add_patch(rect)

    for seg in segments:
        ax.plot(seg.points[:, 0], seg.points[:, 1],
                color='lightgray', linewidth=1.0, alpha=0.5, zorder=1)

    cmap = plt.cm.get_cmap('tab20')
    for idx, path in enumerate(legal_paths):
        color = cmap(idx % 20)
        for seg in path.segments:
            ax.plot(seg.points[:, 0], seg.points[:, 1],
                    color=color, linewidth=2.5, alpha=0.7, zorder=2)
            ax.plot(seg.points[0, 0], seg.points[0, 1],
                    'o', color=color, markersize=6, zorder=3)
            ax.plot(seg.points[-1, 0], seg.points[-1, 1],
                    's', color=color, markersize=6, zorder=3)

    legend_elements = [
        mpatches.Patch(color='lightgray', label='All Segments'),
        mpatches.Patch(color='red', label='Crop Region'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='black', markersize=8, label='Start Point'),
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='black', markersize=8, label='End Point'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Visualization saved to: {out_path}")

    return fig, ax


def visualize_individual_paths(segments: List[LaneSegment],
                               legal_paths: List[LegalPath],
                               crop: CropBox,
                               output_dir: str):
    if plt is None:
        raise RuntimeError("matplotlib is not available for visualization")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Creating individual visualizations in: {output_dir}")

    for path_idx, path in enumerate(legal_paths):
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(crop.xmin - 5, crop.xmax + 5)
        ax.set_ylim(crop.ymin - 5, crop.ymax + 5)
        ax.invert_xaxis()
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        ax.set_title(
            f"Path {path_idx + 1}: {len(path.segments)} segments, {path.total_length:.1f}m total length",
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)

        rect = mpatches.Rectangle(
            (crop.xmin, crop.ymin),
            crop.xmax - crop.xmin,
            crop.ymax - crop.ymin,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            linestyle='--',
            alpha=0.5
        )
        ax.add_patch(rect)

        for seg in segments:
            ax.plot(seg.points[:, 0], seg.points[:, 1],
                    color='lightgray', linewidth=0.8, alpha=0.3, zorder=1)

        cmap = plt.cm.get_cmap('viridis')
        for seg_idx, seg in enumerate(path.segments):
            color = cmap(seg_idx / max(1, len(path.segments) - 1))
            ax.plot(seg.points[:, 0], seg.points[:, 1],
                    color=color, linewidth=3.5, alpha=0.9, zorder=2,
                    label=f"Seg {seg_idx + 1}: road={seg.road_id}, lane={seg.lane_id}")
            ax.plot(seg.points[0, 0], seg.points[0, 1],
                    'o', color=color, markersize=8, zorder=3,
                    markeredgecolor='black', markeredgewidth=1)
            ax.plot(seg.points[-1, 0], seg.points[-1, 1],
                    's', color=color, markersize=8, zorder=3,
                    markeredgecolor='black', markeredgewidth=1)

        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        plt.tight_layout()

        # Name from signature (world-frame)
        sig = build_path_signature(path)
        name = make_path_name(path_idx, sig)
        out_file = os.path.join(output_dir, name + ".png")
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if (path_idx + 1) % 10 == 0:
            print(f"[INFO] Generated {path_idx + 1}/{len(legal_paths)} visualizations")

    print(f"[INFO] All {len(legal_paths)} individual visualizations saved to: {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize all legal path segments in a cropped region"
    )
    parser.add_argument("--nodes", type=str, required=True, help="Path to town nodes JSON file")
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        required=True,
        help="Crop region as: xmin xmax ymin ymax",
    )
    parser.add_argument(
        "--max-yaw-diff",
        type=float,
        default=60.0,
        help="Maximum yaw difference for connecting segments (degrees)",
    )
    parser.add_argument(
        "--connect-radius",
        type=float,
        default=6.0,
        help="Maximum distance to connect segment endpoints (meters)",
    )
    parser.add_argument(
        "--min-path-length",
        type=float,
        default=20.0,
        help="Minimum total path length to display (meters)",
    )
    parser.add_argument("--max-paths", type=int, default=100, help="Maximum number of paths to generate")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum number of segments per path")

    parser.add_argument("--viz", action="store_true", help="Display the visualization")
    parser.add_argument("--out", type=str, default="legal_paths_viz.png", help="Output image file path")

    # NEW:
    parser.add_argument(
        "--out-prompt",
        type=str,
        default="legal_paths_prompt.txt",
        help="Prompt-ready combined file containing all candidate path signatures",
    )
    parser.add_argument(
        "--out-json-detailed",
        type=str,
        default=None,
        help="Optional: also save a pure JSON file containing all candidate path signatures",
    )

    parser.add_argument("--individual", action="store_true", help="Generate separate visualization for each path")
    parser.add_argument(
        "--individual-dir",
        type=str,
        default="legal_paths_individual",
        help="Directory for individual path visualizations (default: legal_paths_individual)",
    )

    args = parser.parse_args()

    crop = CropBox(xmin=args.crop[0], xmax=args.crop[1], ymin=args.crop[2], ymax=args.crop[3])

    print(f"[INFO] Loading nodes from: {args.nodes}")
    data = load_nodes(args.nodes)

    print("[INFO] Building lane segments...")
    all_segments = build_segments(data)
    print(f"[INFO] Total segments in map: {len(all_segments)}")

    print(f"[INFO] Cropping to region: [{crop.xmin}, {crop.xmax}] x [{crop.ymin}, {crop.ymax}]")
    cropped_segments = crop_segments(all_segments, crop)
    print(f"[INFO] Segments in crop region: {len(cropped_segments)}")

    if len(cropped_segments) == 0:
        print("[ERROR] No segments found in crop region!")
        return

    print("[INFO] Building connectivity graph...")
    adj = build_connectivity(
        cropped_segments,
        connect_radius_m=args.connect_radius,
        connect_yaw_tol_deg=args.max_yaw_diff,
    )
    total_connections = sum(len(neighbors) for neighbors in adj)
    print(f"[INFO] Total connections: {total_connections}")

    print("[INFO] Generating legal paths (from outside crop to outside)...")
    legal_paths = generate_legal_paths(
        cropped_segments,
        adj,
        crop,
        min_path_length=args.min_path_length,
        max_paths=args.max_paths,
        max_depth=args.max_depth,
    )

    print(f"[INFO] Found {len(legal_paths)} legal paths")

    if len(legal_paths) == 0:
        print("[WARNING] No legal paths found! Try adjusting parameters:")
        print("  - Increase --max-yaw-diff")
        print("  - Increase --connect-radius")
        print("  - Decrease --min-path-length")
        print("  - Increase --max-paths or --max-depth")
    else:
        lengths = [p.total_length for p in legal_paths]
        seg_counts = [len(p.segments) for p in legal_paths]
        print("[INFO] Path length statistics:")
        print(f"  Min: {min(lengths):.2f} m")
        print(f"  Max: {max(lengths):.2f} m")
        print(f"  Mean: {np.mean(lengths):.2f} m")
        print(f"  Median: {np.median(lengths):.2f} m")
        print("[INFO] Segments per path:")
        print(f"  Min: {min(seg_counts)}")
        print(f"  Max: {max(seg_counts)}")
        print(f"  Mean: {np.mean(seg_counts):.2f}")

    # Build candidate list: name + ideal signature
    params = {
        "max_yaw_diff_deg": args.max_yaw_diff,
        "connect_radius_m": args.connect_radius,
        "min_path_length_m": args.min_path_length,
        "max_paths": args.max_paths,
        "max_depth": args.max_depth,
        "turn_frame": "WORLD_FRAME",  # important!
    }

    candidates: List[Dict[str, Any]] = []
    for i, p in enumerate(legal_paths):
        sig = build_path_signature(p)
        name = make_path_name(i, sig)

        # IMPORTANT: This is purely extra export/logging for --out-json-detailed.
        # It does not affect any generation, matching, naming, or prompting behavior.
        if args.out_json_detailed:
            sig["segments_detailed"] = build_segments_detailed_for_path(p, polyline_sample_n=10)

        candidates.append({"name": name, "signature": sig})

    # Save the combined prompt-ready file (copy/paste directly into LLM)
    save_prompt_file(
        args.out_prompt,
        crop=crop,
        nodes_path=args.nodes,
        params=params,
        paths_named=candidates,
    )

    # Optional: pure JSON version of the same data
    if args.out_json_detailed:
        save_aggregated_signatures_json(
            args.out_json_detailed,
            crop=crop,
            nodes_path=args.nodes,
            params=params,
            paths_named=candidates,
        )

    # Visualize all paths together
    print("[INFO] Creating combined visualization...")
    visualize_legal_paths(cropped_segments, legal_paths, crop, args.out)

    # Individual visualizations
    if args.individual and len(legal_paths) > 0:
        print("[INFO] Generating individual visualizations...")
        visualize_individual_paths(cropped_segments, legal_paths, crop, args.individual_dir)

    if args.viz:
        print("[INFO] Displaying visualization...")
        plt.show()

    print("[INFO] Done!")


if __name__ == "__main__":
    main()
